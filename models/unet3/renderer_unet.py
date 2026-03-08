#!/usr/bin/env python3
"""
U-Net Neural Renderer

U-Net architecture for mask-aware neural rendering with full-image context
via encoder-decoder skip connections. Replaces the shallow fully-convolutional
MaskAwareRenderer which had only an 11x11 receptive field.

Architecture:
    Encoder: 4 blocks (Conv 4x4 stride 2 + InstanceNorm + LeakyReLU)
    Bottleneck: Conv 3x3 + ReLU
    Decoder: 4 blocks (ConvTranspose 4x4 stride 2 + InstanceNorm + ReLU + skip concat)
    Output: Sigmoid + mask skip connection

Input: [B, 7, 1024, 1024] - ref(3) + texture(3) + car_mask(1)
Output: [B, 3, 1024, 1024] - rendered RGB

Usage:
    from models.unet3.renderer_unet import UNetRenderer

    model = UNetRenderer()
    output = model(x)  # [B, 7, 1024, 1024] -> [B, 3, 1024, 1024]

Author: Adversarial Camouflage Research Project
Date: 2026-02-09
"""

import torch
import torch.nn as nn
import numpy as np


class UNetRenderer(nn.Module):
    """
    U-Net mask-aware neural renderer (7-channel input).

    Encoder:
        E1: 7  -> 64  (512x512)
        E2: 64 -> 128 (256x256)
        E3: 128-> 256 (128x128)
        E4: 256-> 512 (64x64)

    Bottleneck:
        512 -> 512 (64x64)

    Decoder (with skip connections):
        D4: 512      -> 256 (128x128), concat E3 -> 512 ch
        D3: 512      -> 128 (256x256), concat E2 -> 256 ch
        D2: 256      -> 64  (512x512), concat E1 -> 128 ch
        D1: 128      -> 3   (1024x1024)

    Output: sigmoid + mask skip connection
    """

    def __init__(self, input_channels=7, resolution=1024):
        super().__init__()
        self.input_channels = input_channels
        self.resolution = resolution

        # Encoder blocks: Conv(4x4, stride=2) + InstanceNorm + LeakyReLU(0.2)
        # E1: no norm on first layer (common U-Net practice)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks: ConvTranspose(4x4, stride=2) + InstanceNorm + ReLU
        # After upsampling, skip connection is concatenated before next block
        self.dec4_up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # After concat with E3 (256ch): 256+256=512 -> process with conv
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec3_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # After concat with E2 (128ch): 128+128=256
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec2_up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # After concat with E1 (64ch): 64+64=128
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec1_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final conv to RGB
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform (matches Keras defaults)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass with mask-based skip connection.

        Background pixels are guaranteed to equal reference:
            output = conv_output * mask + reference * (1 - mask)

        Args:
            x: Input tensor [B, 7, H, W]
               Channels: [ref_R, ref_G, ref_B, tex_R, tex_G, tex_B, mask]

        Returns:
            Rendered image [B, 3, H, W] in range [0, 1]
        """
        reference = x[:, 0:3]  # [B, 3, H, W]
        mask = x[:, 6:7]       # [B, 1, H, W]

        # Encoder
        e1 = self.enc1(x)    # [B, 64, 512, 512]
        e2 = self.enc2(e1)   # [B, 128, 256, 256]
        e3 = self.enc3(e2)   # [B, 256, 128, 128]
        e4 = self.enc4(e3)   # [B, 512, 64, 64]

        # Bottleneck
        b = self.bottleneck(e4)  # [B, 512, 64, 64]

        # Decoder with skip connections
        d4 = self.dec4_up(b)                      # [B, 256, 128, 128]
        d4 = self.dec4_conv(torch.cat([d4, e3], dim=1))  # concat -> [B, 512, 128, 128] -> [B, 256, 128, 128]

        d3 = self.dec3_up(d4)                     # [B, 128, 256, 256]
        d3 = self.dec3_conv(torch.cat([d3, e2], dim=1))  # concat -> [B, 256, 256, 256] -> [B, 128, 256, 256]

        d2 = self.dec2_up(d3)                     # [B, 64, 512, 512]
        d2 = self.dec2_conv(torch.cat([d2, e1], dim=1))  # concat -> [B, 128, 512, 512] -> [B, 64, 512, 512]

        d1 = self.dec1_up(d2)                     # [B, 32, 1024, 1024]

        # Output with sigmoid
        conv_output = torch.sigmoid(self.final_conv(d1))  # [B, 3, 1024, 1024]

        # Mask skip connection: car pixels from network, background from reference
        output = conv_output * mask + reference * (1.0 - mask)

        return output

    def forward_from_components(self, x_ref, texture, mask):
        """
        Convenience method that concatenates inputs before forward pass.

        Args:
            x_ref: Reference image [B, 3, H, W]
            texture: Texture pattern [B, 3, H, W]
            mask: Car mask [B, 1, H, W]

        Returns:
            Rendered image [B, 3, H, W]
        """
        combined = torch.cat([x_ref, texture, mask], dim=1)
        return self.forward(combined)

    def get_model_info(self):
        """Get model metadata."""
        return {
            'framework': 'PyTorch',
            'architecture': 'UNet',
            'input_channels': self.input_channels,
            'resolution': self.resolution,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def load_unet_renderer(model_path, device=None):
    """
    Load a trained U-Net renderer from disk.

    Args:
        model_path: Path to .pt file
        device: Target device, auto-detected if None

    Returns:
        UNetRenderer model in eval mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNetRenderer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    print("=" * 70)
    print("U-NET RENDERER TEST")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Create model
    print("Creating UNetRenderer...")
    model = UNetRenderer()
    model.to(device)

    info = model.get_model_info()
    print("Model Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    # Test forward pass
    print("Testing forward pass...")
    x = torch.randn(1, 7, 1024, 1024, device=device)
    print(f"  Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    assert output.shape == (1, 3, 1024, 1024), f"Wrong shape: {output.shape}"
    print()

    # Test gradient flow
    print("Testing gradient flow through texture...")
    x_ref = torch.randn(1, 3, 1024, 1024, device=device)
    texture = torch.randn(1, 3, 1024, 1024, device=device, requires_grad=True)
    mask = torch.ones(1, 1, 1024, 1024, device=device)  # All car pixels

    output = model.forward_from_components(x_ref, texture, mask)
    loss = output.mean()
    loss.backward()

    grad_norm = texture.grad.norm().item()
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")

    if grad_norm > 0:
        print("  PASSED: Gradients flow correctly through texture input")
    else:
        print("  FAILED: No gradients flowing to texture!")
    print()

    # Test background preservation
    print("Testing background preservation (mask=0 regions)...")
    x_ref = torch.rand(1, 3, 1024, 1024, device=device)
    texture = torch.rand(1, 3, 1024, 1024, device=device)
    mask = torch.zeros(1, 1, 1024, 1024, device=device)  # All background

    with torch.no_grad():
        output = model.forward_from_components(x_ref, texture, mask)

    diff = (output - x_ref).abs().max().item()
    print(f"  Max diff from reference (should be 0): {diff:.8f}")
    if diff < 1e-6:
        print("  PASSED: Background perfectly preserved")
    else:
        print("  FAILED: Background not preserved!")
    print()

    if device == 'cuda':
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB peak")

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
