#!/usr/bin/env python3
"""
PyTorch Renderer Gradient Flow Test

Verifies that gradients flow correctly through the PyTorch neural renderer:
    texture -> renderer -> output -> loss -> texture.grad

This is a critical test for the adversarial attack pipeline.

Usage:
    python test_scripts/test_renderer_pytorch.py

Expected output:
    - Gradients exist (texture.grad is not None)
    - Gradient norm > 0
    - All tests pass

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from models.renderer.renderer_pytorch import MaskAwareRenderer


def test_renderer_forward_pass():
    """Test basic forward pass works correctly."""
    print("Test 1: Basic forward pass")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    # Create random input [batch, 7, H, W] in valid image range [0, 1]
    batch_size = 2
    resolution = 1024
    x = torch.rand(batch_size, 7, resolution, resolution, device=device)

    with torch.no_grad():
        output = model(x)

    # Check output shape
    assert output.shape == (batch_size, 3, resolution, resolution), \
        f"Expected shape {(batch_size, 3, resolution, resolution)}, got {output.shape}"

    # Check output range (should be [0, 1] due to sigmoid)
    assert output.min() >= 0.0, f"Output min {output.min()} < 0"
    assert output.max() <= 1.0, f"Output max {output.max()} > 1"

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("  PASSED")
    print()
    return True


def test_gradient_flow_through_texture():
    """Test that gradients flow from loss to texture input."""
    print("Test 2: Gradient flow through texture")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()  # Even in eval mode, gradients should flow

    # Create inputs
    resolution = 1024
    x_ref = torch.randn(1, 3, resolution, resolution, device=device)
    texture = torch.randn(1, 3, resolution, resolution, device=device, requires_grad=True)
    mask = torch.randn(1, 1, resolution, resolution, device=device)

    # Forward pass using forward_from_components
    output = model.forward_from_components(x_ref, texture, mask)

    # Compute loss (simple mean)
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert texture.grad is not None, "Gradients are None!"

    grad_norm = texture.grad.norm().item()
    assert grad_norm > 0, f"Gradient norm is 0!"

    print(f"  Loss: {loss.item():.6f}")
    print(f"  texture.grad exists: {texture.grad is not None}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Gradient shape: {texture.grad.shape}")
    print("  PASSED: Gradients flow correctly through texture!")
    print()
    return True


def test_gradient_flow_through_combined_input():
    """Test gradient flow with concatenated 7-channel input."""
    print("Test 3: Gradient flow through combined input")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    resolution = 1024

    # Create texture with gradient tracking
    texture = torch.randn(1, 3, resolution, resolution, device=device, requires_grad=True)

    # Other inputs (no gradients needed)
    x_ref = torch.randn(1, 3, resolution, resolution, device=device)
    mask = torch.randn(1, 1, resolution, resolution, device=device)

    # Manually concatenate (as would happen in pipeline)
    combined = torch.cat([x_ref, texture, mask], dim=1)  # [1, 7, H, W]

    # Forward pass
    output = model(combined)

    # Loss
    loss = output.mean()

    # Backward
    loss.backward()

    # Check
    assert texture.grad is not None, "Gradients are None!"
    grad_norm = texture.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    print(f"  Combined input shape: {combined.shape}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print("  PASSED: Gradients flow through concatenated input!")
    print()
    return True


def test_gradient_flow_with_upsampling():
    """Test gradients flow through texture upsampling (critical for coarse texture)."""
    print("Test 4: Gradient flow through upsampling")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    # Coarse texture (128x128 as used in training)
    coarse_size = 128
    full_res = 1024

    texture_coarse = torch.randn(1, 3, coarse_size, coarse_size, device=device, requires_grad=True)

    # Upsample to full resolution
    texture_full = F.interpolate(
        texture_coarse,
        size=(full_res, full_res),
        mode='bicubic',
        align_corners=False
    )

    # Create other inputs
    x_ref = torch.randn(1, 3, full_res, full_res, device=device)
    mask = torch.randn(1, 1, full_res, full_res, device=device)

    # Forward pass
    output = model.forward_from_components(x_ref, texture_full, mask)

    # Loss
    loss = output.mean()

    # Backward
    loss.backward()

    # Check gradients on COARSE texture
    assert texture_coarse.grad is not None, "Gradients are None!"
    grad_norm = texture_coarse.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    print(f"  Coarse texture shape: {texture_coarse.shape}")
    print(f"  Upsampled texture shape: {texture_full.shape}")
    print(f"  Gradient norm on coarse: {grad_norm:.6f}")
    print("  PASSED: Gradients flow through upsampling!")
    print()
    return True


def test_batch_gradient_flow():
    """Test gradient flow with batch processing (6 viewpoints)."""
    print("Test 5: Batch gradient flow (6 viewpoints)")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    batch_size = 6  # EOT viewpoints
    coarse_size = 128
    full_res = 1024

    # Single coarse texture
    texture_coarse = torch.randn(1, 3, coarse_size, coarse_size, device=device, requires_grad=True)

    # Upsample
    texture_full = F.interpolate(
        texture_coarse,
        size=(full_res, full_res),
        mode='bicubic',
        align_corners=False
    )

    # Tile for all viewpoints
    texture_batch = texture_full.expand(batch_size, -1, -1, -1)

    # Batch inputs
    x_ref_batch = torch.randn(batch_size, 3, full_res, full_res, device=device)
    mask_batch = torch.randn(batch_size, 1, full_res, full_res, device=device)

    # Forward pass
    output_batch = model.forward_from_components(x_ref_batch, texture_batch, mask_batch)

    # Loss (mean over batch - simulates EOT)
    loss = output_batch.mean()

    # Backward
    loss.backward()

    # Check
    assert texture_coarse.grad is not None, "Gradients are None!"
    grad_norm = texture_coarse.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    print(f"  Batch size: {batch_size}")
    print(f"  Output batch shape: {output_batch.shape}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print("  PASSED: Gradients flow through batch processing!")
    print()
    return True


def test_background_preservation():
    """Test that background pixels exactly match reference image."""
    print("Test 7: Background preservation (skip connection)")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    resolution = 1024

    # Create inputs with a clear binary mask
    x_ref = torch.rand(1, 3, resolution, resolution, device=device)
    texture = torch.rand(1, 3, resolution, resolution, device=device)
    # Binary mask: top half = car (1), bottom half = background (0)
    mask = torch.zeros(1, 1, resolution, resolution, device=device)
    mask[:, :, :resolution // 2, :] = 1.0

    with torch.no_grad():
        output = model.forward_from_components(x_ref, texture, mask)

    # Background region (bottom half where mask=0) should exactly match reference
    bg_output = output[:, :, resolution // 2:, :]
    bg_reference = x_ref[:, :, resolution // 2:, :]

    max_diff = (bg_output - bg_reference).abs().max().item()

    print(f"  Max background diff: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Background not preserved! Max diff: {max_diff}"

    print("  PASSED: Background pixels exactly match reference!")
    print()
    return True


def test_gradient_flow_with_skip_connection():
    """Test that gradients still flow to texture through the masked path."""
    print("Test 8: Gradient flow with skip connection")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    resolution = 1024

    x_ref = torch.rand(1, 3, resolution, resolution, device=device)
    texture = torch.rand(1, 3, resolution, resolution, device=device, requires_grad=True)
    # Partial mask: some car pixels, some background
    mask = torch.zeros(1, 1, resolution, resolution, device=device)
    mask[:, :, :resolution // 2, :] = 1.0

    output = model.forward_from_components(x_ref, texture, mask)

    # Loss only on car region (where mask=1)
    car_output = output[:, :, :resolution // 2, :]
    loss = car_output.mean()
    loss.backward()

    assert texture.grad is not None, "Gradients are None!"
    grad_norm = texture.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    # Gradients should be non-zero in masked (car) region input channels
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print("  PASSED: Gradients flow through skip connection!")
    print()
    return True


def test_gradient_magnitude():
    """Verify gradients have reasonable magnitude (not vanishing/exploding)."""
    print("Test 6: Gradient magnitude check")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskAwareRenderer()
    model.to(device)
    model.eval()

    resolution = 1024

    # Multiple runs to check consistency
    grad_norms = []
    for i in range(5):
        texture = torch.randn(1, 3, resolution, resolution, device=device, requires_grad=True)
        x_ref = torch.randn(1, 3, resolution, resolution, device=device)
        mask = torch.randn(1, 1, resolution, resolution, device=device)

        output = model.forward_from_components(x_ref, texture, mask)
        loss = output.mean()
        loss.backward()

        grad_norms.append(texture.grad.norm().item())

    mean_grad = np.mean(grad_norms)
    std_grad = np.std(grad_norms)

    print(f"  Gradient norms over 5 runs: {grad_norms}")
    print(f"  Mean: {mean_grad:.6f}")
    print(f"  Std: {std_grad:.6f}")

    # Check not vanishing (< 1e-8) or exploding (> 1e8)
    assert mean_grad > 1e-8, f"Gradients may be vanishing (mean={mean_grad})"
    assert mean_grad < 1e8, f"Gradients may be exploding (mean={mean_grad})"

    print("  PASSED: Gradients have reasonable magnitude!")
    print()
    return True


def main():
    print("=" * 70)
    print("PYTORCH RENDERER GRADIENT FLOW TEST")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
        test_renderer_forward_pass,
        test_gradient_flow_through_texture,
        test_gradient_flow_through_combined_input,
        test_gradient_flow_with_upsampling,
        test_batch_gradient_flow,
        test_gradient_magnitude,
        test_background_preservation,
        test_gradient_flow_with_skip_connection,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            failed += 1
            print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("ALL TESTS PASSED - Renderer is fully differentiable!")
        return 0
    else:
        print("SOME TESTS FAILED - Check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
