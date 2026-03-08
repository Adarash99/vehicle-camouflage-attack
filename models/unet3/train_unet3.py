#!/usr/bin/env python3
"""
U-Net Neural Renderer Training Script (L1 + VGG Perceptual Loss, Multi-Dataset)

Trains the U-Net renderer using mask-weighted L1 loss combined with
VGG perceptual loss. Supports multiple training/validation datasets
via ConcatDataset for combined solid-color + multi-color training.

Usage:
    python models/unet3/train_unet3.py \
        --datasets dataset_8k_revised/train dataset_multicolor/train \
        --val-datasets dataset_8k_revised/val dataset_multicolor/val \
        --epochs 100

    # Resume from checkpoint
    python models/unet3/train_unet3.py \
        --datasets dataset_8k_revised/train dataset_multicolor/train \
        --resume models/unet3/trained/checkpoints/epoch_050.pt

    # Quick test (3 epochs)
    python models/unet3/train_unet3.py \
        --datasets dataset_8k_revised/train --val-datasets dataset_8k_revised/val --test

Date: 2026-02-10
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models

from models.unet3.renderer_unet import UNetRenderer
from models.unet3.renderer_dataset import RendererDataset


class MaskWeightedL1Loss(nn.Module):
    """L1 loss with separate weights for car and background regions."""

    def __init__(self, car_weight=1.0, bg_weight=0.1):
        super().__init__()
        self.car_weight = car_weight
        self.bg_weight = bg_weight

    def forward(self, pred, target, mask):
        diff = torch.abs(pred - target)

        car_pixels = mask.sum() * 3
        if car_pixels > 0:
            car_loss = (diff * mask).sum() / car_pixels
        else:
            car_loss = torch.tensor(0.0, device=pred.device)

        bg_mask = 1.0 - mask
        bg_pixels = bg_mask.sum() * 3
        if bg_pixels > 0:
            bg_loss = (diff * bg_mask).sum() / bg_pixels
        else:
            bg_loss = torch.tensor(0.0, device=pred.device)

        return self.car_weight * car_loss + self.bg_weight * bg_loss


class VGGPerceptualLoss(nn.Module):
    """VGG16-based perceptual loss. Inputs downsampled to 256x256."""

    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(vgg.features[:4]))   # -> relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features[4:9]))  # -> relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features[9:16])) # -> relu3_3
        self.slice4 = nn.Sequential(*list(vgg.features[16:23]))# -> relu4_3

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _extract_features(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self._normalize(x)

        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        f4 = self.slice4(f3)
        return [f1, f2, f3, f4]

    def forward(self, pred, target):
        pred_feats = self._extract_features(pred)
        target_feats = self._extract_features(target)

        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.l1_loss(pf, tf)
        return loss / len(pred_feats)


class CombinedLoss(nn.Module):
    """Combined L1 + VGG perceptual loss."""

    def __init__(self, lambda_perceptual=0.1, car_weight=1.0, bg_weight=0.1):
        super().__init__()
        self.l1_loss = MaskWeightedL1Loss(car_weight=car_weight, bg_weight=bg_weight)
        self.vgg_loss = VGGPerceptualLoss()
        self.lambda_perceptual = lambda_perceptual

    def forward(self, pred, target, mask):
        l1 = self.l1_loss(pred, target, mask)
        vgg = self.vgg_loss(pred, target)
        total = l1 + self.lambda_perceptual * vgg
        return total, l1, vgg


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_vgg = 0.0
    num_batches = len(train_loader)
    use_amp = scaler is not None

    for batch_idx, (x_combined, y_target) in enumerate(train_loader):
        x_combined = x_combined.to(device)
        y_target = y_target.to(device)

        optimizer.zero_grad()

        mask = x_combined[:, 6:7]

        if use_amp:
            with autocast():
                output = model(x_combined)
                loss, l1_val, vgg_val = criterion(output, y_target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x_combined)
            loss, l1_val, vgg_val = criterion(output, y_target, mask)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_l1 += l1_val.item()
        total_vgg += vgg_val.item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            avg_loss = total_loss / (batch_idx + 1)
            avg_l1 = total_l1 / (batch_idx + 1)
            avg_vgg = total_vgg / (batch_idx + 1)
            print(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                  f"Loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, VGG: {avg_vgg:.6f})")

    n = num_batches
    return total_loss / n, total_l1 / n, total_vgg / n


def validate(model, val_loader, criterion, device, use_amp=False):
    """Validate the model with optional AMP."""
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_vgg = 0.0

    with torch.no_grad():
        for x_combined, y_target in val_loader:
            x_combined = x_combined.to(device)
            y_target = y_target.to(device)

            mask = x_combined[:, 6:7]

            if use_amp:
                with autocast():
                    output = model(x_combined)
                    loss, l1_val, vgg_val = criterion(output, y_target, mask)
            else:
                output = model(x_combined)
                loss, l1_val, vgg_val = criterion(output, y_target, mask)
            total_loss += loss.item()
            total_l1 += l1_val.item()
            total_vgg += vgg_val.item()

    n = len(val_loader)
    return total_loss / n, total_l1 / n, total_vgg / n


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, scaler=None):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def train_unet3(
    dataset_paths,
    output_dir='models/unet3/trained/',
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    lambda_perceptual=0.1,
    val_dataset_paths=None,
    val_split=0.1,
    resume_path=None,
    num_workers=4,
    device=None,
    use_amp=True
):
    """
    Train the U-Net neural renderer with L1 + VGG perceptual loss on multiple datasets.

    Args:
        dataset_paths: List of paths to training datasets
        output_dir: Directory to save model and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial Adam learning rate
        lambda_perceptual: Weight for VGG perceptual loss
        val_dataset_paths: List of paths to validation datasets
        val_split: Fraction for validation if no val_dataset_paths
        resume_path: Path to checkpoint to resume from
        num_workers: Data loading workers
        device: Training device
        use_amp: Use automatic mixed precision
    """
    print("=" * 70)
    print("U-NET NEURAL RENDERER TRAINING (L1 + Perceptual, Multi-Dataset)")
    print("=" * 70)
    print()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Create data loaders with ConcatDataset
    print("Loading datasets...")
    print(f"  Training datasets: {dataset_paths}")

    train_datasets = [RendererDataset(p) for p in dataset_paths]
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    for i, ds in enumerate(train_datasets):
        print(f"    [{i}] {dataset_paths[i]}: {len(ds)} samples")
    print(f"  Total train samples: {len(train_dataset)}")

    if val_dataset_paths:
        print(f"  Validation datasets: {val_dataset_paths}")
        val_datasets = [RendererDataset(p) for p in val_dataset_paths]
        val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        for i, ds in enumerate(val_datasets):
            print(f"    [{i}] {val_dataset_paths[i]}: {len(ds)} samples")
        print(f"  Total val samples: {len(val_dataset)}")
    else:
        # Fallback: use first dataset with split
        from models.unet3.renderer_dataset import create_data_loaders
        train_loader, val_loader = create_data_loaders(
            dataset_paths[0], batch_size=batch_size, val_split=val_split,
            num_workers=num_workers
        )
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    print()

    # Create model
    print("Creating U-Net model...")
    model = UNetRenderer()
    model.to(device)
    info = model.get_model_info()
    print(f"  Architecture: {info['architecture']}")
    print(f"  Total params: {info['total_params']:,}")
    print()

    # Loss function and optimizer
    print(f"Loss: L1 (mask-weighted) + {lambda_perceptual} * VGG Perceptual")
    criterion = CombinedLoss(lambda_perceptual=lambda_perceptual, car_weight=1.0, bg_weight=0.1)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # AMP scaler
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    if scaler:
        print(f"Mixed Precision: Enabled (fp16)")
    else:
        print(f"Mixed Precision: Disabled")
    print()

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_path:
        print(f"Resuming from: {resume_path}")
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, resume_path, device, scaler
        )
        start_epoch += 1
        print(f"  Resumed at epoch {start_epoch}, best loss: {best_val_loss:.6f}")
        print()

    # Training history
    history = {
        'train_loss': [],
        'train_l1': [],
        'train_vgg': [],
        'val_loss': [],
        'val_l1': [],
        'val_vgg': [],
        'lr': [],
    }

    # Save config
    config = {
        'architecture': 'UNetRenderer',
        'loss': 'L1 + VGG Perceptual',
        'lambda_perceptual': lambda_perceptual,
        'dataset_paths': [str(p) for p in dataset_paths],
        'val_dataset_paths': [str(p) for p in val_dataset_paths] if val_dataset_paths else None,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device),
        'use_amp': scaler is not None,
        'total_params': info['total_params'],
        'start_time': datetime.now().isoformat(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("Starting training...")
    print("-" * 70)

    training_start_time = datetime.now()

    for epoch in range(start_epoch, epochs):
        train_loss, train_l1, train_vgg = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        val_loss, val_l1, val_vgg = validate(
            model, val_loader, criterion, device, use_amp=(scaler is not None)
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_l1'].append(train_l1)
        history['train_vgg'].append(train_vgg)
        history['val_loss'].append(val_loss)
        history['val_l1'].append(val_l1)
        history['val_vgg'].append(val_vgg)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.6f} (L1: {train_l1:.6f}, VGG: {train_vgg:.6f}) | "
              f"Val: {val_loss:.6f} (L1: {val_l1:.6f}, VGG: {val_vgg:.6f}) | "
              f"LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  -> New best model saved (val_loss: {val_loss:.6f})")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                checkpoint_dir / f'epoch_{epoch+1:03d}.pt', scaler
            )

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')

    # Training time
    training_end_time = datetime.now()
    total_time = training_end_time - training_start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Save history
    history['total_training_time_seconds'] = total_time.total_seconds()
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best validation loss (L1 + Perceptual): {best_val_loss:.6f}")
    print(f"  Final model: {output_dir / 'final_model.pt'}")
    print(f"  Best model: {output_dir / 'best_model.pt'}")
    print(f"  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 70)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train U-Net Neural Renderer (L1 + Perceptual, Multi-Dataset)')

    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Paths to training datasets')
    parser.add_argument('--val-datasets', nargs='+', default=None,
                        help='Paths to validation datasets')
    parser.add_argument('--output', type=str, default='models/unet3/trained/',
                        help='Output directory for model and logs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (4 recommended for U-Net on 24GB GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda-perceptual', type=float, default=0.1,
                        help='Weight for VGG perceptual loss (default: 0.1)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (fallback if no --val-datasets)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test with 3 epochs')

    args = parser.parse_args()

    if args.test:
        print("Running quick test (3 epochs)...")
        args.epochs = 3
        args.batch_size = 2

    use_amp = args.amp and not args.no_amp

    train_unet3(
        dataset_paths=args.datasets,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_perceptual=args.lambda_perceptual,
        val_dataset_paths=args.val_datasets,
        val_split=args.val_split,
        resume_path=args.resume,
        num_workers=args.workers,
        device=args.device,
        use_amp=use_amp
    )


if __name__ == '__main__':
    main()
