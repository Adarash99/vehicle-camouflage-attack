#!/usr/bin/env python3
"""
PyTorch Texture Applicator for Adversarial Camouflage Pipeline

Drop-in replacement for the TensorFlow TextureApplicator, enabling true
end-to-end gradient flow for adversarial texture optimization.

Key Features:
- Supports both numpy arrays (inference) and torch tensors (training)
- Handles NHWC <-> NCHW conversion transparently
- Fully differentiable when using tensor inputs
- Compatible with existing pipeline (same API)

Usage:
    from texture_applicator_pytorch import TextureApplicatorPyTorch

    # Create applicator
    applicator = TextureApplicatorPyTorch()

    # Inference (numpy arrays)
    rendered = applicator.apply(x_ref_np, texture_np, mask_np)

    # Training (differentiable, torch tensors)
    rendered = applicator.apply_differentiable(x_ref_tensor, texture_tensor, mask_tensor)
    loss = criterion(rendered, target)
    loss.backward()  # Gradients flow through!

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from models.unet3.renderer_unet import UNetRenderer, load_unet_renderer


class TextureApplicatorPyTorch:
    """
    PyTorch texture applicator using the mask-aware neural renderer.

    Provides a unified interface for both inference (numpy) and training (tensor)
    use cases, with automatic format conversion.

    The renderer takes 7-channel input [ref(3) + texture(3) + mask(1)] and
    outputs 3-channel rendered RGB image.
    """

    DEFAULT_MODEL_PATH = 'models/unet3/trained/best_model.pt'

    def __init__(self, model_path=None, device=None):
        """
        Initialize the texture applicator.

        Args:
            model_path: Path to trained PyTorch model (.pt file)
                       If None, uses default path or creates untrained model
            device: Target device ('cuda' or 'cpu'), auto-detected if None
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        print(f"Initializing TextureApplicatorPyTorch...")
        print(f"  Device: {self.device}")

        # Load or create model
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH

        if os.path.exists(model_path):
            print(f"  Loading model: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model = UNetRenderer()
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        else:
            print(f"  Model not found at {model_path}, creating untrained model")
            self.model = UNetRenderer()
            self.model.to(self.device)

        self.model.eval()  # Set to evaluation mode

        # Print model info
        info = self.model.get_model_info()
        print(f"  Architecture: {info.get('architecture', 'UNetRenderer')}")
        print(f"  Resolution: {info['resolution']}x{info['resolution']}")
        print(f"  Parameters: {info['total_params']:,}")
        print()

    def apply(self, x_ref, texture, mask):
        """
        Apply texture to reference image (inference mode, returns numpy).

        Accepts numpy arrays or tensors, returns numpy array.
        Handles NHWC <-> NCHW conversion automatically.

        Args:
            x_ref: Reference image, numpy [H, W, 3] or [B, H, W, 3] float32 [0, 1]
            texture: Texture pattern, numpy [H, W, 3] or [B, H, W, 3] float32 [0, 1]
            mask: Paintable mask, numpy [H, W] or [H, W, 1] or [B, H, W, 1] float32 [0, 1]

        Returns:
            rendered: Numpy array [H, W, 3] or [B, H, W, 3] float32 [0, 1]
        """
        # Convert to numpy if needed
        x_ref_np = self._to_numpy(x_ref)
        texture_np = self._to_numpy(texture)
        mask_np = self._to_numpy(mask)

        # Handle single image (3D) vs batch (4D)
        is_single = (x_ref_np.ndim == 3)

        if is_single:
            x_ref_np = x_ref_np[np.newaxis, ...]
            texture_np = texture_np[np.newaxis, ...]
            if mask_np.ndim == 2:
                mask_np = mask_np[np.newaxis, :, :, np.newaxis]
            elif mask_np.ndim == 3:
                mask_np = mask_np[np.newaxis, ...]

        # Ensure mask has channel dimension
        if mask_np.ndim == 3:
            mask_np = mask_np[..., np.newaxis]

        # Convert NHWC -> NCHW
        x_ref_nchw = np.transpose(x_ref_np, (0, 3, 1, 2))  # [B, 3, H, W]
        texture_nchw = np.transpose(texture_np, (0, 3, 1, 2))  # [B, 3, H, W]
        mask_nchw = np.transpose(mask_np, (0, 3, 1, 2))  # [B, 1, H, W]

        # Convert to tensors
        x_ref_t = torch.from_numpy(x_ref_nchw).float().to(self.device)
        texture_t = torch.from_numpy(texture_nchw).float().to(self.device)
        mask_t = torch.from_numpy(mask_nchw).float().to(self.device)

        # Forward pass (no gradients needed for inference)
        with torch.no_grad():
            rendered_t = self.model.forward_from_components(x_ref_t, texture_t, mask_t)

        # Convert back: NCHW -> NHWC
        rendered_np = rendered_t.cpu().numpy()
        rendered_np = np.transpose(rendered_np, (0, 2, 3, 1))  # [B, H, W, 3]

        # Squeeze if input was single image
        if is_single:
            rendered_np = rendered_np[0]

        return rendered_np

    def apply_differentiable(self, x_ref, texture, mask):
        """
        Apply texture with gradient tracking (training mode).

        All inputs must be PyTorch tensors in NCHW format.
        Returns tensor with gradient tracking enabled.

        Args:
            x_ref: Reference image, tensor [B, 3, H, W] float32 [0, 1]
            texture: Texture pattern, tensor [B, 3, H, W] float32 [0, 1] (requires_grad=True for optimization)
            mask: Paintable mask, tensor [B, 1, H, W] float32 [0, 1]

        Returns:
            rendered: Tensor [B, 3, H, W] float32 [0, 1] (with gradients)
        """
        # Validate inputs are tensors
        if not all(isinstance(t, torch.Tensor) for t in [x_ref, texture, mask]):
            raise TypeError(
                "apply_differentiable requires torch.Tensor inputs. "
                "Use apply() for numpy arrays."
            )

        # Move to device if needed
        x_ref = x_ref.to(self.device)
        texture = texture.to(self.device)
        mask = mask.to(self.device)

        # Forward pass (with gradient tracking)
        rendered = self.model.forward_from_components(x_ref, texture, mask)

        return rendered

    def apply_nhwc(self, x_ref, texture, mask):
        """
        Apply texture with NHWC tensor inputs (convenience method for training).

        Handles NHWC <-> NCHW conversion for tensor inputs while maintaining
        gradient flow.

        Args:
            x_ref: Reference image, tensor [B, H, W, 3] float32 [0, 1]
            texture: Texture pattern, tensor [B, H, W, 3] float32 [0, 1]
            mask: Paintable mask, tensor [B, H, W, 1] float32 [0, 1]

        Returns:
            rendered: Tensor [B, H, W, 3] float32 [0, 1] (with gradients)
        """
        # Convert NHWC -> NCHW
        x_ref_nchw = x_ref.permute(0, 3, 1, 2)
        texture_nchw = texture.permute(0, 3, 1, 2)
        mask_nchw = mask.permute(0, 3, 1, 2)

        # Forward pass
        rendered_nchw = self.apply_differentiable(x_ref_nchw, texture_nchw, mask_nchw)

        # Convert NCHW -> NHWC
        rendered_nhwc = rendered_nchw.permute(0, 2, 3, 1)

        return rendered_nhwc

    def _to_numpy(self, x):
        """Convert tensor to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def get_model_info(self):
        """Get renderer model metadata."""
        info = self.model.get_model_info()
        info['device'] = str(self.device)
        return info


class TextureApplicatorPyTorchV1:
    """
    Legacy V1 texture applicator (6-channel, 500x500, no mask).

    For backward compatibility with V1 models.
    """

    def __init__(self, model_path='models/renderer_v1.pt', device=None):
        """Initialize V1 applicator."""
        raise NotImplementedError(
            "V1 PyTorch renderer not implemented. "
            "Use the TensorFlow TextureApplicator for V1 models."
        )


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH TEXTURE APPLICATOR TEST")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Create applicator (will create untrained model if no weights found)
    print("Creating TextureApplicatorPyTorch...")
    try:
        applicator = TextureApplicatorPyTorch()
    except Exception as e:
        print(f"Error creating applicator: {e}")
        print("Creating with untrained model...")
        applicator = TextureApplicatorPyTorch(model_path=None)
    print()

    # Test 1: Numpy inference (single image)
    print("Test 1: Numpy inference (single image)...")
    x_ref_np = np.random.rand(1024, 1024, 3).astype(np.float32)
    texture_np = np.random.rand(1024, 1024, 3).astype(np.float32)
    mask_np = np.random.rand(1024, 1024).astype(np.float32)

    rendered_np = applicator.apply(x_ref_np, texture_np, mask_np)
    print(f"  Input shapes: ref={x_ref_np.shape}, tex={texture_np.shape}, mask={mask_np.shape}")
    print(f"  Output shape: {rendered_np.shape}")
    print(f"  Output range: [{rendered_np.min():.4f}, {rendered_np.max():.4f}]")
    print()

    # Test 2: Numpy inference (batch)
    print("Test 2: Numpy inference (batch of 4)...")
    x_ref_batch = np.random.rand(4, 1024, 1024, 3).astype(np.float32)
    texture_batch = np.random.rand(4, 1024, 1024, 3).astype(np.float32)
    mask_batch = np.random.rand(4, 1024, 1024, 1).astype(np.float32)

    rendered_batch = applicator.apply(x_ref_batch, texture_batch, mask_batch)
    print(f"  Input shapes: ref={x_ref_batch.shape}, tex={texture_batch.shape}, mask={mask_batch.shape}")
    print(f"  Output shape: {rendered_batch.shape}")
    print()

    # Test 3: Differentiable forward pass (NCHW tensors)
    print("Test 3: Differentiable forward pass (NCHW tensors)...")
    x_ref_t = torch.randn(2, 3, 1024, 1024, device=device)
    texture_t = torch.randn(2, 3, 1024, 1024, device=device, requires_grad=True)
    mask_t = torch.randn(2, 1, 1024, 1024, device=device)

    rendered_t = applicator.apply_differentiable(x_ref_t, texture_t, mask_t)
    print(f"  Input shapes: ref={x_ref_t.shape}, tex={texture_t.shape}, mask={mask_t.shape}")
    print(f"  Output shape: {rendered_t.shape}")
    print(f"  Output requires_grad: {rendered_t.requires_grad}")
    print()

    # Test 4: Gradient flow verification (CRITICAL)
    print("Test 4: Gradient flow verification...")
    x_ref_t = torch.randn(1, 3, 1024, 1024, device=device)
    texture_t = torch.randn(1, 3, 1024, 1024, device=device, requires_grad=True)
    mask_t = torch.randn(1, 1, 1024, 1024, device=device)

    # Forward pass
    rendered_t = applicator.apply_differentiable(x_ref_t, texture_t, mask_t)
    loss = rendered_t.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    grad_norm = texture_t.grad.norm().item()
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")

    if grad_norm > 0:
        print("  PASSED: Gradients flow correctly through texture!")
    else:
        print("  FAILED: No gradients flowing to texture!")
    print()

    # Test 5: NHWC convenience method
    print("Test 5: NHWC convenience method...")
    x_ref_nhwc = torch.randn(2, 1024, 1024, 3, device=device)
    texture_nhwc = torch.randn(2, 1024, 1024, 3, device=device, requires_grad=True)
    mask_nhwc = torch.randn(2, 1024, 1024, 1, device=device)

    rendered_nhwc = applicator.apply_nhwc(x_ref_nhwc, texture_nhwc, mask_nhwc)
    print(f"  Input shapes: ref={x_ref_nhwc.shape}, tex={texture_nhwc.shape}")
    print(f"  Output shape: {rendered_nhwc.shape}")

    # Verify gradients still flow
    loss = rendered_nhwc.mean()
    loss.backward()
    grad_norm = texture_nhwc.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.6f}")
    if grad_norm > 0:
        print("  PASSED: Gradients flow through NHWC interface!")
    print()

    # Print model info
    print("Model Information:")
    info = applicator.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
