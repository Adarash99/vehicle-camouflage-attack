#!/usr/bin/env python3
"""
PyTorch Dataset for Neural Renderer Training

Loads reference/texture/mask/rendered image quadruplets for training
the mask-aware neural renderer.

Usage:
    from models.unet3.renderer_dataset import RendererDataset

    dataset = RendererDataset('dataset/')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for x_combined, y_target in dataloader:
        # x_combined: [batch, 7, H, W] - ref(3) + texture(3) + mask(1)
        # y_target: [batch, 3, H, W] - rendered ground truth
        output = model(x_combined)
        loss = criterion(output, y_target)

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob


class RendererDataset(Dataset):
    """
    PyTorch Dataset for training the mask-aware neural renderer.

    Expected directory structure:
        dataset_path/
        ├── reference/   (gray car on black, 1024x1024 PNG)
        ├── texture/     (texture pattern on black, 1024x1024 PNG)
        ├── rendered/    (CARLA-rendered result, 1024x1024 PNG)
        └── mask/        (binary paintable mask, 1024x1024 PNG)

    All images should have matching filenames (e.g., 0.png, 1.png, ...).

    Output format (PyTorch NCHW convention):
        x_combined: [7, H, W] - channels: [ref_R, ref_G, ref_B, tex_R, tex_G, tex_B, mask]
        y_target: [3, H, W] - rendered ground truth RGB
    """

    def __init__(self, dataset_path, resolution=1024, transform=None):
        """
        Initialize the dataset.

        Args:
            dataset_path: Path to dataset directory
            resolution: Expected image resolution (default: 1024)
            transform: Optional torchvision transforms to apply
        """
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution
        self.transform = transform

        # Verify required directories exist
        required_dirs = ['reference', 'texture', 'rendered', 'mask']
        for d in required_dirs:
            path = self.dataset_path / d
            if not path.exists():
                raise FileNotFoundError(f"Required directory not found: {path}")

        # Get all sample IDs from reference directory
        ref_files = sorted(glob(str(self.dataset_path / 'reference' / '*.png')))
        self.sample_ids = [Path(f).stem for f in ref_files]

        if len(self.sample_ids) == 0:
            raise ValueError(f"No PNG files found in {self.dataset_path / 'reference'}")

        print(f"RendererDataset initialized:")
        print(f"  Path: {self.dataset_path}")
        print(f"  Samples: {len(self.sample_ids)}")
        print(f"  Resolution: {resolution}x{resolution}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Load a single sample.

        Returns:
            x_combined: torch.Tensor [7, H, W] float32 in [0, 1]
            y_target: torch.Tensor [3, H, W] float32 in [0, 1]
        """
        sample_id = self.sample_ids[idx]

        # Load reference image
        ref_path = self.dataset_path / 'reference' / f'{sample_id}.png'
        ref = self._load_image(ref_path)  # [H, W, 3] RGB float32

        # Load texture image
        tex_path = self.dataset_path / 'texture' / f'{sample_id}.png'
        tex = self._load_image(tex_path)  # [H, W, 3] RGB float32

        # Load mask (grayscale)
        mask_path = self.dataset_path / 'mask' / f'{sample_id}.png'
        mask = self._load_mask(mask_path)  # [H, W, 1] float32

        # Load rendered target
        rendered_path = self.dataset_path / 'rendered' / f'{sample_id}.png'
        rendered = self._load_image(rendered_path)  # [H, W, 3] RGB float32

        # Combine inputs: [ref(3) + texture(3) + mask(1)] -> [7, H, W]
        # Note: Convert from NHWC to NCHW (PyTorch convention)
        x_combined = np.concatenate([ref, tex, mask], axis=-1)  # [H, W, 7]
        x_combined = np.transpose(x_combined, (2, 0, 1))  # [7, H, W]

        # Target: [3, H, W]
        y_target = np.transpose(rendered, (2, 0, 1))  # [3, H, W]

        # Convert to tensors
        x_combined = torch.from_numpy(x_combined).float()
        y_target = torch.from_numpy(y_target).float()

        # Apply transforms if provided
        if self.transform is not None:
            x_combined = self.transform(x_combined)
            y_target = self.transform(y_target)

        return x_combined, y_target

    def _load_image(self, path):
        """Load RGB image and normalize to [0, 1]."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Verify resolution
        if img.shape[:2] != (self.resolution, self.resolution):
            raise ValueError(
                f"Image resolution mismatch: expected ({self.resolution}, {self.resolution}), "
                f"got {img.shape[:2]} for {path}"
            )

        return img

    def _load_mask(self, path):
        """Load grayscale mask and normalize to [0, 1]."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {path}")

        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0

        # Add channel dimension [H, W] -> [H, W, 1]
        mask = mask[:, :, np.newaxis]

        # Verify resolution
        if mask.shape[:2] != (self.resolution, self.resolution):
            raise ValueError(
                f"Mask resolution mismatch: expected ({self.resolution}, {self.resolution}), "
                f"got {mask.shape[:2]} for {path}"
            )

        return mask

    def get_sample_paths(self, idx):
        """Get file paths for a sample (for debugging)."""
        sample_id = self.sample_ids[idx]
        return {
            'reference': self.dataset_path / 'reference' / f'{sample_id}.png',
            'texture': self.dataset_path / 'texture' / f'{sample_id}.png',
            'mask': self.dataset_path / 'mask' / f'{sample_id}.png',
            'rendered': self.dataset_path / 'rendered' / f'{sample_id}.png',
        }


class RendererDatasetV1(Dataset):
    """
    Legacy dataset for V1 renderer (6-channel, 500x500, no mask).

    For backward compatibility with V1 models.
    """

    def __init__(self, dataset_path, resolution=500):
        """Initialize V1 dataset."""
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution

        # V1 doesn't require mask directory
        required_dirs = ['reference', 'texture', 'rendered']
        for d in required_dirs:
            path = self.dataset_path / d
            if not path.exists():
                raise FileNotFoundError(f"Required directory not found: {path}")

        ref_files = sorted(glob(str(self.dataset_path / 'reference' / '*.png')))
        self.sample_ids = [Path(f).stem for f in ref_files]

        print(f"RendererDatasetV1 initialized: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """Load V1 sample (no mask)."""
        sample_id = self.sample_ids[idx]

        # Load images
        ref = self._load_image(self.dataset_path / 'reference' / f'{sample_id}.png')
        tex = self._load_image(self.dataset_path / 'texture' / f'{sample_id}.png')
        rendered = self._load_image(self.dataset_path / 'rendered' / f'{sample_id}.png')

        # Combine inputs: [ref(3) + texture(3)] -> [6, H, W]
        x_combined = np.concatenate([ref, tex], axis=-1)  # [H, W, 6]
        x_combined = np.transpose(x_combined, (2, 0, 1))  # [6, H, W]

        y_target = np.transpose(rendered, (2, 0, 1))  # [3, H, W]

        return torch.from_numpy(x_combined).float(), torch.from_numpy(y_target).float()

    def _load_image(self, path):
        """Load and normalize image."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to load: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0


def create_data_loaders(dataset_path, batch_size=8, val_split=0.1, num_workers=4):
    """
    Create train and validation data loaders.

    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    dataset = RendererDataset(dataset_path)

    # Split into train/val
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset split: {n_train} train, {n_val} validation")

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH RENDERER DATASET TEST")
    print("=" * 70)
    print()

    # Check if dataset exists
    dataset_path = 'dataset/'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Creating mock dataset for testing...")

        # Create mock dataset
        os.makedirs(f'{dataset_path}/reference', exist_ok=True)
        os.makedirs(f'{dataset_path}/texture', exist_ok=True)
        os.makedirs(f'{dataset_path}/rendered', exist_ok=True)
        os.makedirs(f'{dataset_path}/mask', exist_ok=True)

        for i in range(5):
            # Create dummy images (small for testing)
            dummy = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            mask = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)

            cv2.imwrite(f'{dataset_path}/reference/{i}.png', dummy)
            cv2.imwrite(f'{dataset_path}/texture/{i}.png', dummy)
            cv2.imwrite(f'{dataset_path}/rendered/{i}.png', dummy)
            cv2.imwrite(f'{dataset_path}/mask/{i}.png', mask)

        print("Mock dataset created")
        print()

    # Test dataset
    print("Testing RendererDataset...")
    try:
        dataset = RendererDataset(dataset_path, resolution=1024)
        print(f"  Samples: {len(dataset)}")

        # Load first sample
        x, y = dataset[0]
        print(f"  x_combined shape: {x.shape}")
        print(f"  y_target shape: {y.shape}")
        print(f"  x dtype: {x.dtype}")
        print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
        print()

        # Test DataLoader
        print("Testing DataLoader...")
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        print(f"  Batch x shape: {batch_x.shape}")
        print(f"  Batch y shape: {batch_y.shape}")
        print()

        # Test create_data_loaders if enough samples
        if len(dataset) >= 10:
            print("Testing train/val split...")
            train_loader, val_loader = create_data_loaders(dataset_path, batch_size=2)
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"  Error: {e}")
        print("  (This is expected if dataset is incomplete)")

    print()
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)
