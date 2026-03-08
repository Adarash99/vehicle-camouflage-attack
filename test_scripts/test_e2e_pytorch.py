#!/usr/bin/env python3
"""
End-to-End PyTorch Pipeline Gradient Flow Test

Verifies that gradients flow correctly through the ENTIRE adversarial pipeline:
    texture -> renderer -> detector -> loss -> texture.grad

This is the ultimate test for the PyTorch migration - it ensures that
true backpropagation works from the attack loss all the way back to
the texture parameters.

Usage:
    python test_scripts/test_e2e_pytorch.py

Expected output:
    - Gradients flow from loss to texture
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
from attack.detector_pytorch import EfficientDetPyTorch
from attack.loss_pytorch import attack_loss_pytorch


def test_e2e_gradient_flow_basic():
    """Test basic end-to-end gradient flow: texture -> renderer -> detector -> loss."""
    print("Test 1: Basic E2E gradient flow")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create renderer
    print("  Loading renderer...")
    renderer = MaskAwareRenderer()
    renderer.to(device)
    renderer.eval()

    # Create detector
    print("  Loading detector...")
    detector = EfficientDetPyTorch(device=device)

    # Create inputs
    resolution = 1024
    detector_size = 512

    # Texture with gradient tracking
    texture = torch.randn(1, 3, resolution, resolution, device=device, requires_grad=True)

    # Reference and mask (no gradients)
    x_ref = torch.randn(1, 3, resolution, resolution, device=device)
    mask = torch.randn(1, 1, resolution, resolution, device=device)

    print("  Running forward pass...")

    # Step 1: Render
    rendered = renderer.forward_from_components(x_ref, texture, mask)
    print(f"    Rendered shape: {rendered.shape}")

    # Step 2: Resize for detector
    rendered_resized = F.interpolate(
        rendered,
        size=(detector_size, detector_size),
        mode='bilinear',
        align_corners=False
    )
    print(f"    Resized shape: {rendered_resized.shape}")

    # Step 3: Detect (WITH gradients!)
    class_logits, box_preds = detector.forward_pre_nms_with_grad(rendered_resized)
    print(f"    Class logits shape: {class_logits.shape}")

    # Step 4: Compute attack loss
    loss = attack_loss_pytorch(class_logits, car_class_id=2)
    print(f"    Loss: {loss.item():.6f}")

    # Step 5: Backward pass
    print("  Running backward pass...")
    loss.backward()

    # Verify gradients
    assert texture.grad is not None, "texture.grad is None!"
    grad_norm = texture.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    print(f"    texture.grad exists: True")
    print(f"    Gradient norm: {grad_norm:.6f}")
    print("  PASSED: End-to-end gradients flow correctly!")
    print()
    return True


def test_e2e_gradient_flow_with_upsampling():
    """Test E2E with coarse texture upsampling (actual training setup)."""
    print("Test 2: E2E with coarse texture upsampling")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create renderer
    renderer = MaskAwareRenderer()
    renderer.to(device)
    renderer.eval()

    # Create detector
    detector = EfficientDetPyTorch(device=device)

    # Coarse texture (128x128 as used in training)
    coarse_size = 128
    full_res = 1024
    detector_size = 512

    texture_coarse = torch.randn(1, 3, coarse_size, coarse_size, device=device, requires_grad=True)

    # Upsample to full resolution
    texture_full = F.interpolate(
        texture_coarse,
        size=(full_res, full_res),
        mode='bicubic',
        align_corners=False
    )

    # Reference and mask
    x_ref = torch.randn(1, 3, full_res, full_res, device=device)
    mask = torch.randn(1, 1, full_res, full_res, device=device)

    # Forward through renderer
    rendered = renderer.forward_from_components(x_ref, texture_full, mask)

    # Resize and detect
    rendered_resized = F.interpolate(
        rendered, size=(detector_size, detector_size),
        mode='bilinear', align_corners=False
    )
    class_logits, _ = detector.forward_pre_nms_with_grad(rendered_resized)

    # Loss
    loss = attack_loss_pytorch(class_logits, car_class_id=2)

    # Backward
    loss.backward()

    # Verify gradients on COARSE texture
    assert texture_coarse.grad is not None, "texture_coarse.grad is None!"
    grad_norm = texture_coarse.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    print(f"  Coarse texture: {texture_coarse.shape}")
    print(f"  Upsampled: {texture_full.shape}")
    print(f"  Gradient norm on coarse: {grad_norm:.6f}")
    print("  PASSED: Gradients flow through upsampling!")
    print()
    return True


def test_e2e_batch_processing():
    """Test E2E with batch (6 viewpoints for EOT)."""
    print("Test 3: E2E batch processing (6 viewpoints)")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create renderer
    renderer = MaskAwareRenderer()
    renderer.to(device)
    renderer.eval()

    # Create detector
    detector = EfficientDetPyTorch(device=device)

    batch_size = 6  # EOT viewpoints
    coarse_size = 128
    full_res = 1024
    detector_size = 512

    # Single coarse texture
    texture_coarse = torch.randn(1, 3, coarse_size, coarse_size, device=device, requires_grad=True)

    # Upsample
    texture_full = F.interpolate(
        texture_coarse, size=(full_res, full_res),
        mode='bicubic', align_corners=False
    )

    # Tile for all viewpoints
    texture_batch = texture_full.expand(batch_size, -1, -1, -1)

    # Batch reference and mask
    x_ref_batch = torch.randn(batch_size, 3, full_res, full_res, device=device)
    mask_batch = torch.randn(batch_size, 1, full_res, full_res, device=device)

    # Forward through renderer (batch)
    rendered_batch = renderer.forward_from_components(x_ref_batch, texture_batch, mask_batch)

    # Resize and detect
    rendered_resized = F.interpolate(
        rendered_batch, size=(detector_size, detector_size),
        mode='bilinear', align_corners=False
    )
    class_logits, _ = detector.forward_pre_nms_with_grad(rendered_resized)

    # Loss (mean over batch = EOT expectation)
    loss = attack_loss_pytorch(class_logits, car_class_id=2)

    # Backward
    loss.backward()

    # Verify
    assert texture_coarse.grad is not None, "Gradients are None!"
    grad_norm = texture_coarse.grad.norm().item()
    assert grad_norm > 0, "Gradient norm is 0!"

    print(f"  Batch size: {batch_size}")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print("  PASSED: Batch E2E gradients work correctly!")
    print()
    return True


def test_optimization_step():
    """Test that a full optimization step works (gradient -> update)."""
    print("Test 4: Full optimization step")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create renderer
    renderer = MaskAwareRenderer()
    renderer.to(device)
    renderer.eval()

    # Create detector
    detector = EfficientDetPyTorch(device=device)

    coarse_size = 128
    full_res = 1024
    detector_size = 512

    # Texture parameter
    texture = torch.randn(1, 3, coarse_size, coarse_size, device=device, requires_grad=True)
    texture_initial = texture.clone().detach()

    # Optimizer
    optimizer = torch.optim.Adam([texture], lr=0.01)

    # Fixed inputs
    x_ref = torch.randn(1, 3, full_res, full_res, device=device)
    mask = torch.randn(1, 1, full_res, full_res, device=device)

    print("  Running 3 optimization steps...")

    losses = []
    for step in range(3):
        optimizer.zero_grad()

        # Forward
        texture_full = F.interpolate(
            texture, size=(full_res, full_res),
            mode='bicubic', align_corners=False
        )
        rendered = renderer.forward_from_components(x_ref, texture_full, mask)
        rendered_resized = F.interpolate(
            rendered, size=(detector_size, detector_size),
            mode='bilinear', align_corners=False
        )
        class_logits, _ = detector.forward_pre_nms_with_grad(rendered_resized)
        loss = attack_loss_pytorch(class_logits, car_class_id=2)

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        losses.append(loss.item())
        print(f"    Step {step}: loss = {loss.item():.6f}")

    # Verify texture changed
    texture_diff = (texture - texture_initial).abs().mean().item()
    assert texture_diff > 0, "Texture didn't change after optimization!"

    print(f"  Texture change: {texture_diff:.6f}")
    print("  PASSED: Optimization steps work correctly!")
    print()
    return True


def test_gradient_not_vanishing():
    """Verify gradients don't vanish through the pipeline."""
    print("Test 5: Gradient not vanishing")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    renderer = MaskAwareRenderer()
    renderer.to(device)
    renderer.eval()

    detector = EfficientDetPyTorch(device=device)

    coarse_size = 128
    full_res = 1024
    detector_size = 512

    grad_norms = []

    for i in range(5):
        texture = torch.randn(1, 3, coarse_size, coarse_size, device=device, requires_grad=True)
        x_ref = torch.randn(1, 3, full_res, full_res, device=device)
        mask = torch.randn(1, 1, full_res, full_res, device=device)

        # Forward
        texture_full = F.interpolate(
            texture, size=(full_res, full_res),
            mode='bicubic', align_corners=False
        )
        rendered = renderer.forward_from_components(x_ref, texture_full, mask)
        rendered_resized = F.interpolate(
            rendered, size=(detector_size, detector_size),
            mode='bilinear', align_corners=False
        )
        class_logits, _ = detector.forward_pre_nms_with_grad(rendered_resized)
        loss = attack_loss_pytorch(class_logits, car_class_id=2)

        # Backward
        loss.backward()

        grad_norms.append(texture.grad.norm().item())

    mean_grad = np.mean(grad_norms)
    min_grad = np.min(grad_norms)

    print(f"  Gradient norms: {[f'{g:.6f}' for g in grad_norms]}")
    print(f"  Mean: {mean_grad:.6f}")
    print(f"  Min: {min_grad:.6f}")

    assert min_grad > 1e-10, f"Gradients may be vanishing (min={min_grad})"
    print("  PASSED: Gradients not vanishing!")
    print()
    return True


def main():
    print("=" * 70)
    print("END-TO-END PYTORCH PIPELINE GRADIENT FLOW TEST")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    print("This test verifies the ENTIRE adversarial attack pipeline:")
    print("  texture -> renderer -> detector -> loss -> texture.grad")
    print()

    tests = [
        test_e2e_gradient_flow_basic,
        test_e2e_gradient_flow_with_upsampling,
        test_e2e_batch_processing,
        test_optimization_step,
        test_gradient_not_vanishing,
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
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("ALL TESTS PASSED!")
        print()
        print("The PyTorch pipeline is FULLY DIFFERENTIABLE!")
        print("True backpropagation works from loss -> texture.")
        print("No more finite differences needed!")
        return 0
    else:
        print("SOME TESTS FAILED - Check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
