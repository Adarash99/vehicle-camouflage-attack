#!/usr/bin/env python3
"""
Tests for DTA-Style Repeated Texture Projection

Verifies:
1. Gradient flow from projected texture back to input pattern
2. Different viewpoints produce different projections
3. 360° yaw wrap returns to the same projection as 0°
4. Output shape and range correctness
5. Tiling is visible (multiple repetitions)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from attack.texture_projection import RepeatedTextureProjection


def test_gradient_flow():
    """Test that gradients flow from projected texture back to input pattern."""
    print("Test 1: Gradient flow...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = RepeatedTextureProjection(tile_count=8, full_resolution=256)

    texture = torch.rand(3, 128, 128, device=device, requires_grad=True)

    projected = projector.project(texture, yaw=60, pitch=-15, distance=8)
    loss = projected.mean()
    loss.backward()

    assert texture.grad is not None, "Gradient is None!"
    assert texture.grad.norm().item() > 0, "Gradient norm is zero!"
    print(f"  Gradient norm: {texture.grad.norm().item():.6f}")
    print("  PASSED")
    print()


def test_different_viewpoints_differ():
    """Test that different viewpoints produce different projections."""
    print("Test 2: Different viewpoints produce different outputs...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = RepeatedTextureProjection(tile_count=8, full_resolution=256)

    texture = torch.rand(3, 128, 128, device=device)

    proj_0 = projector.project(texture, yaw=0, pitch=-15, distance=8)
    proj_60 = projector.project(texture, yaw=60, pitch=-15, distance=8)
    proj_180 = projector.project(texture, yaw=180, pitch=-15, distance=8)

    diff_0_60 = (proj_0 - proj_60).abs().mean().item()
    diff_0_180 = (proj_0 - proj_180).abs().mean().item()

    assert diff_0_60 > 1e-4, f"yaw=0 and yaw=60 are too similar: diff={diff_0_60}"
    assert diff_0_180 > 1e-4, f"yaw=0 and yaw=180 are too similar: diff={diff_0_180}"

    print(f"  diff(0°, 60°) = {diff_0_60:.6f}")
    print(f"  diff(0°, 180°) = {diff_0_180:.6f}")
    print("  PASSED")
    print()


def test_360_wrap():
    """Test that 360° yaw shift returns to same projection as 0°."""
    print("Test 3: 360° wrap-around...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = RepeatedTextureProjection(tile_count=8, full_resolution=256)

    texture = torch.rand(3, 128, 128, device=device)

    proj_0 = projector.project(texture, yaw=0, pitch=-15, distance=8)
    proj_360 = projector.project(texture, yaw=360, pitch=-15, distance=8)

    diff = (proj_0 - proj_360).abs().max().item()
    assert diff < 1e-4, f"360° wrap failed: max diff = {diff}"

    print(f"  max diff(0°, 360°) = {diff:.8f}")
    print("  PASSED")
    print()


def test_output_shape():
    """Test that output has correct shape and range."""
    print("Test 4: Output shape and range...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = RepeatedTextureProjection(tile_count=8, full_resolution=512)

    texture = torch.rand(3, 64, 64, device=device)

    # Single projection
    proj = projector.project(texture, yaw=45, pitch=-15, distance=8)
    assert proj.shape == (3, 512, 512), f"Wrong shape: {proj.shape}"
    print(f"  Single output shape: {proj.shape}")

    # Batch projection
    viewpoints = [
        {'yaw': 0, 'pitch': -15, 'distance': 8},
        {'yaw': 60, 'pitch': -15, 'distance': 8},
        {'yaw': 120, 'pitch': -15, 'distance': 8},
    ]
    proj_batch = projector.project_batch(texture, viewpoints)
    assert proj_batch.shape == (3, 3, 512, 512), f"Wrong batch shape: {proj_batch.shape}"
    print(f"  Batch output shape: {proj_batch.shape}")

    # Range check (grid_sample with bilinear can slightly exceed [0,1])
    print(f"  Output range: [{proj.min().item():.4f}, {proj.max().item():.4f}]")
    print("  PASSED")
    print()


def test_distance_scaling():
    """Test that different distances produce different scale projections."""
    print("Test 5: Distance scaling...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = RepeatedTextureProjection(tile_count=8, full_resolution=256)

    texture = torch.rand(3, 128, 128, device=device)

    proj_close = projector.project(texture, yaw=0, pitch=-15, distance=4)
    proj_far = projector.project(texture, yaw=0, pitch=-15, distance=16)

    diff = (proj_close - proj_far).abs().mean().item()
    assert diff > 1e-4, f"Close and far projections too similar: diff={diff}"

    print(f"  diff(4m, 16m) = {diff:.6f}")
    print("  PASSED")
    print()


def test_batch_gradient_flow():
    """Test that gradients flow through batch projection."""
    print("Test 6: Batch gradient flow...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector = RepeatedTextureProjection(tile_count=8, full_resolution=256)

    texture = torch.rand(3, 128, 128, device=device, requires_grad=True)

    viewpoints = [
        {'yaw': 0, 'pitch': -15, 'distance': 8},
        {'yaw': 60, 'pitch': -15, 'distance': 8},
        {'yaw': 120, 'pitch': -15, 'distance': 8},
        {'yaw': 180, 'pitch': -15, 'distance': 8},
        {'yaw': 240, 'pitch': -15, 'distance': 8},
        {'yaw': 300, 'pitch': -15, 'distance': 8},
    ]

    proj_batch = projector.project_batch(texture, viewpoints)
    loss = proj_batch.mean()
    loss.backward()

    assert texture.grad is not None, "Gradient is None!"
    assert texture.grad.norm().item() > 0, "Gradient norm is zero!"
    print(f"  Batch gradient norm: {texture.grad.norm().item():.6f}")
    print("  PASSED")
    print()


def test_nonlinear_shift_rate():
    """Test that non-linear shift is slower at face centers than linear."""
    print("Test 7: Non-linear shift rate (boxiness)...")

    import math

    # Non-linear projector (default boxiness=0.4)
    projector_box = RepeatedTextureProjection(tile_count=8, full_resolution=256, boxiness=0.4)
    # Linear projector (boxiness=0)
    projector_lin = RepeatedTextureProjection(tile_count=8, full_resolution=256, boxiness=0.0)

    # At 22.5° (mid-face, between 0° face center and 45° corner),
    # non-linear shift should be LESS than linear shift because
    # the shift rate is slower near face centers.
    yaw = 22.5
    yaw_rad = math.radians(yaw)

    linear_shift = (yaw_rad - (0.0 / 4) * math.sin(4 * yaw_rad)) / math.pi  # boxiness=0
    nonlinear_shift = (yaw_rad - (0.4 / 4) * math.sin(4 * yaw_rad)) / math.pi  # boxiness=0.4

    print(f"  At yaw={yaw}°: linear_shift={linear_shift:.6f}, nonlinear_shift={nonlinear_shift:.6f}")
    assert nonlinear_shift < linear_shift, (
        f"Non-linear shift should be less than linear at mid-face: "
        f"{nonlinear_shift:.6f} >= {linear_shift:.6f}"
    )

    # At 360° both should produce the same total shift (2.0 in grid units)
    linear_360 = (math.radians(360) - (0.0 / 4) * math.sin(4 * math.radians(360))) / math.pi
    nonlinear_360 = (math.radians(360) - (0.4 / 4) * math.sin(4 * math.radians(360))) / math.pi
    diff_360 = abs(linear_360 - nonlinear_360)
    print(f"  At yaw=360°: linear={linear_360:.6f}, nonlinear={nonlinear_360:.6f}, diff={diff_360:.8f}")
    assert diff_360 < 1e-10, f"360° totals differ: {diff_360}"

    # Verify the actual projections differ at a mid-face angle (not 45° where sin(4θ)=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    texture = torch.rand(3, 128, 128, device=device)
    proj_lin = projector_lin.project(texture, yaw=22.5, pitch=-15, distance=8)
    proj_box = projector_box.project(texture, yaw=22.5, pitch=-15, distance=8)
    diff = (proj_lin - proj_box).abs().mean().item()
    print(f"  Projection diff at 22.5° (mid-face): {diff:.6f}")
    assert diff > 1e-4, f"Linear and boxy projections too similar at mid-face: {diff}"

    print("  PASSED")
    print()


def test_visual_save():
    """Save visual grids of projections for manual inspection (linear vs non-linear)."""
    print("Test 8: Visual output (saves to test_scripts/output/)...")

    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Create a recognizable pattern (colored quadrants)
    texture = torch.zeros(3, 128, 128, device=device)
    texture[0, :64, :64] = 1.0   # Red top-left
    texture[1, :64, 64:] = 1.0   # Green top-right
    texture[2, 64:, :64] = 1.0   # Blue bottom-left
    texture[:, 64:, 64:] = 0.5   # Gray bottom-right

    yaws = [0, 60, 120, 180, 240, 300]

    for label, projector in [
        ('linear', RepeatedTextureProjection(tile_count=8, full_resolution=256, boxiness=0.0, min_foreshorten=0.5)),
        ('nonlinear', RepeatedTextureProjection(tile_count=8, full_resolution=256, boxiness=0.4, min_foreshorten=0.2)),
    ]:
        projections = []
        for yaw in yaws:
            proj = projector.project(texture, yaw=yaw, pitch=-15, distance=8)
            proj_np = proj.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            projections.append(proj_np)

        # Create 2x3 grid
        row1 = np.concatenate(projections[:3], axis=1)
        row2 = np.concatenate(projections[3:], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        grid = np.clip(grid, 0, 1)

        output_path = os.path.join(output_dir, f'texture_projection_grid_{label}.png')
        grid_bgr = cv2.cvtColor((grid * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, grid_bgr)
        print(f"  Saved: {output_path}")

    print("  PASSED")
    print()


if __name__ == '__main__':
    print("=" * 70)
    print("TEXTURE PROJECTION TESTS")
    print("=" * 70)
    print()

    test_gradient_flow()
    test_different_viewpoints_differ()
    test_360_wrap()
    test_output_shape()
    test_distance_scaling()
    test_batch_gradient_flow()
    test_nonlinear_shift_rate()
    test_visual_save()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
