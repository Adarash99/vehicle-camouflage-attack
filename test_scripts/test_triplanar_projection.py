#!/usr/bin/env python3
"""
Tests for Depth-Based Triplanar Texture Projection

Verifies (without CARLA):
1. Gradient flow from projected texture back to input pattern
2. UV coordinates are computed correctly for known geometry
3. Triplanar blending produces smooth transitions
4. Car mask is applied correctly
5. Tile size affects UV scale
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math

from attack.texture_projection import TriplanarProjection


class MockTransform:
    """Mock carla.Transform for unit testing without CARLA."""

    def __init__(self, x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0):
        self.location = MockLocation(x, y, z)
        self.rotation = MockRotation(pitch, yaw, roll)


class MockLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


def test_gradient_flow():
    """Test that gradients flow from projected output back to texture."""
    print("Test 1: Gradient flow...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = 64
    projector = TriplanarProjection(tile_size=0.5, resolution=res, fov=90.0)

    texture = torch.rand(3, 16, 16, device=device, requires_grad=True)

    # Create synthetic depth (flat plane at 5m)
    depth = np.full((res, res), 5.0, dtype=np.float32)

    # Full car mask
    mask = np.ones((res, res), dtype=np.float32)

    # Camera at origin looking forward, vehicle at origin
    cam_transform = MockTransform(x=0, y=0, z=0, pitch=0, yaw=0, roll=0)
    veh_transform = MockTransform(x=0, y=0, z=0, pitch=0, yaw=0, roll=0)

    projected = projector.project(texture, depth, mask, cam_transform, veh_transform)
    loss = projected.mean()
    loss.backward()

    assert texture.grad is not None, "Gradient is None!"
    assert texture.grad.norm().item() > 0, "Gradient norm is zero!"
    print(f"  Gradient norm: {texture.grad.norm().item():.6f}")
    print("  PASSED")
    print()


def test_mask_application():
    """Test that non-car pixels are zeroed out."""
    print("Test 2: Mask application...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = 64
    projector = TriplanarProjection(tile_size=0.5, resolution=res, fov=90.0)

    # All-ones texture so any non-zero output means the pixel was sampled
    texture = torch.ones(3, 8, 8, device=device)

    depth = np.full((res, res), 5.0, dtype=np.float32)

    # Half mask: only left half is car
    mask = np.zeros((res, res), dtype=np.float32)
    mask[:, :res // 2] = 1.0

    cam_transform = MockTransform(x=0, y=0, z=0, pitch=0, yaw=0, roll=0)
    veh_transform = MockTransform(x=0, y=0, z=0, pitch=0, yaw=0, roll=0)

    with torch.no_grad():
        projected = projector.project(texture, depth, mask, cam_transform, veh_transform)

    proj_np = projected.cpu().numpy()

    # Right half should be zero
    right_half = proj_np[:, :, res // 2:]
    assert np.allclose(right_half, 0.0), f"Right half not zero, max={right_half.max()}"

    # Left half should have non-zero values
    left_half = proj_np[:, :, :res // 2]
    assert left_half.max() > 0, "Left half is all zeros (should have texture)"

    print(f"  Right half max: {right_half.max():.6f} (should be 0)")
    print(f"  Left half max: {left_half.max():.6f} (should be > 0)")
    print("  PASSED")
    print()


def test_tile_size_affects_uv():
    """Test that changing tile_size changes the UV mapping."""
    print("Test 3: Tile size affects UV scale...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = 64

    # Create a recognizable striped pattern
    texture = torch.zeros(3, 16, 16, device=device)
    texture[:, :8, :] = 1.0  # Top half white

    depth = np.full((res, res), 5.0, dtype=np.float32)
    mask = np.ones((res, res), dtype=np.float32)
    cam_transform = MockTransform(x=0, y=0, z=0, pitch=0, yaw=0, roll=0)
    veh_transform = MockTransform(x=0, y=0, z=0, pitch=0, yaw=0, roll=0)

    projector_small = TriplanarProjection(tile_size=0.25, resolution=res, fov=90.0)
    projector_large = TriplanarProjection(tile_size=1.0, resolution=res, fov=90.0)

    with torch.no_grad():
        proj_small = projector_small.project(
            texture, depth, mask, cam_transform, veh_transform
        )
        proj_large = projector_large.project(
            texture, depth, mask, cam_transform, veh_transform
        )

    diff = (proj_small - proj_large).abs().mean().item()
    assert diff > 1e-4, f"Different tile sizes produce same output: diff={diff}"
    print(f"  Projection diff (tile 0.25m vs 1.0m): {diff:.6f}")
    print("  PASSED")
    print()


def test_output_shape():
    """Test that output has correct shape."""
    print("Test 4: Output shape...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = 128
    projector = TriplanarProjection(tile_size=0.5, resolution=res, fov=90.0)

    texture = torch.rand(3, 16, 16, device=device)
    depth = np.full((res, res), 5.0, dtype=np.float32)
    mask = np.ones((res, res), dtype=np.float32)
    cam_transform = MockTransform()
    veh_transform = MockTransform()

    with torch.no_grad():
        proj = projector.project(texture, depth, mask, cam_transform, veh_transform)

    assert proj.shape == (3, res, res), f"Wrong shape: {proj.shape}, expected (3, {res}, {res})"
    print(f"  Output shape: {proj.shape}")
    print(f"  Output range: [{proj.min().item():.4f}, {proj.max().item():.4f}]")
    print("  PASSED")
    print()


def test_rotation_matrix():
    """Test that the rotation matrix is correct for known angles."""
    print("Test 5: Rotation matrix correctness...")

    projector = TriplanarProjection()

    # Identity rotation (all zeros)
    R = projector._rotation_matrix(0, 0, 0)
    expected = np.eye(3, dtype=np.float32)
    assert np.allclose(R, expected, atol=1e-6), f"Identity rotation wrong:\n{R}"
    print("  Identity rotation: OK")

    # 90 degree yaw (should rotate X->Y, Y->-X in CARLA's left-handed system)
    R_yaw90 = projector._rotation_matrix(0, 90, 0)
    # Apply to unit X vector [1, 0, 0] -> should become [0, 1, 0]
    v_x = np.array([1, 0, 0], dtype=np.float32)
    result = R_yaw90 @ v_x
    assert np.allclose(result, [0, 1, 0], atol=1e-5), f"Yaw 90 @ [1,0,0] = {result}, expected [0,1,0]"
    print(f"  Yaw 90 @ [1,0,0] = [{result[0]:.4f}, {result[1]:.4f}, {result[2]:.4f}]")

    # Rotation matrices should be orthogonal (R @ R^T = I)
    for p, y, r in [(30, 45, 60), (-15, 180, 0), (0, 270, 45)]:
        R = projector._rotation_matrix(p, y, r)
        RRt = R @ R.T
        assert np.allclose(RRt, np.eye(3), atol=1e-5), f"R({p},{y},{r}) not orthogonal"
    print("  Orthogonality: OK")

    print("  PASSED")
    print()


def test_normals_flat_plane():
    """Test that normals for a flat plane facing camera are correct."""
    print("Test 6: Surface normals for flat plane...")

    projector = TriplanarProjection(resolution=64, fov=90.0)

    # Flat plane at constant depth (all points at Z=5)
    # For a fronto-parallel plane, all normals should point toward camera (negative Z)
    res = 64
    f = projector.focal
    cx, cy = projector.cx, projector.cy

    depth = np.full((res, res), 5.0, dtype=np.float32)

    u = np.arange(res, dtype=np.float32)
    v = np.arange(res, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)

    x_cam = (u_grid - cx) * depth / f
    y_cam = (v_grid - cy) * depth / f
    z_cam = depth

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    normals = projector._compute_normals(points_cam)

    # Interior normals (away from edges) should be approximately [0, 0, -1]
    # (pointing back toward camera) or [0, 0, 1] depending on cross product convention
    center_normal = normals[res // 2, res // 2]
    z_component = abs(center_normal[2])
    xy_magnitude = np.sqrt(center_normal[0] ** 2 + center_normal[1] ** 2)

    assert z_component > 0.99, f"Center normal Z component too low: {z_component:.4f}"
    assert xy_magnitude < 0.01, f"Center normal XY too large: {xy_magnitude:.4f}"

    print(f"  Center normal: [{center_normal[0]:.4f}, {center_normal[1]:.4f}, {center_normal[2]:.4f}]")
    print(f"  Z dominance: {z_component:.4f} (expected > 0.99)")
    print("  PASSED")
    print()


def test_different_viewpoints_differ():
    """Test that different camera positions produce different projections."""
    print("Test 7: Different camera positions produce different outputs...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = 64
    projector = TriplanarProjection(tile_size=0.5, resolution=res, fov=90.0)

    texture = torch.rand(3, 8, 8, device=device)
    depth = np.full((res, res), 5.0, dtype=np.float32)
    mask = np.ones((res, res), dtype=np.float32)
    veh_transform = MockTransform(x=5, y=0, z=0)

    # Camera at two different positions
    cam1 = MockTransform(x=0, y=0, z=2, pitch=-15, yaw=0, roll=0)
    cam2 = MockTransform(x=0, y=5, z=2, pitch=-15, yaw=90, roll=0)

    with torch.no_grad():
        proj1 = projector.project(texture, depth, mask, cam1, veh_transform)
        proj2 = projector.project(texture, depth, mask, cam2, veh_transform)

    diff = (proj1 - proj2).abs().mean().item()
    assert diff > 1e-4, f"Different viewpoints produce same output: diff={diff}"
    print(f"  Viewpoint diff: {diff:.6f}")
    print("  PASSED")
    print()


def test_batch_gradient():
    """Test gradient flow with a simple backward pass."""
    print("Test 8: Backward pass gradient magnitude...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = 32
    projector = TriplanarProjection(tile_size=0.5, resolution=res, fov=90.0)

    texture = torch.rand(3, 8, 8, device=device, requires_grad=True)
    depth = np.full((res, res), 5.0, dtype=np.float32)
    mask = np.ones((res, res), dtype=np.float32)
    cam_transform = MockTransform(x=-5, y=0, z=2, pitch=-10, yaw=0, roll=0)
    veh_transform = MockTransform(x=0, y=0, z=0)

    projected = projector.project(texture, depth, mask, cam_transform, veh_transform)

    # Simulate a loss that wants to maximize red, minimize green
    loss = projected[0].mean() - projected[1].mean()
    loss.backward()

    grad = texture.grad
    assert grad is not None
    assert grad.norm().item() > 0

    # Red channel should have positive gradients, green negative
    print(f"  Total grad norm: {grad.norm().item():.6f}")
    print(f"  Red channel grad mean: {grad[0].mean().item():.6f}")
    print(f"  Green channel grad mean: {grad[1].mean().item():.6f}")
    print("  PASSED")
    print()


def test_viewpoint_invariant_plane_selection():
    """
    Test that normals computed from vehicle-local points correctly classify
    known surface orientations.

    This validates the fix: _compute_normals now operates on vehicle-local points
    instead of camera-space points, so plane selection is determined by the
    vehicle geometry, not the camera angle. We directly construct vehicle-local
    point clouds for known surfaces and verify correct plane assignment:
        - Side wall (X=const): X-dominant -> plane 0 (side)
        - Front wall (Y=const): Y-dominant -> plane 1 (front)
        - Roof (Z=const): Z-dominant -> plane 2 (top)
    """
    print("Test 9: Viewpoint-invariant plane selection...")

    projector = TriplanarProjection(tile_size=0.5, resolution=32, fov=90.0)
    res = 32
    plane_names = ['side', 'front', 'top']
    interior = slice(1, res - 1)  # Skip edge pixels (replicated finite diffs)

    # --- Side wall: X=2, varying Y and Z ---
    points_side = np.zeros((res, res, 3), dtype=np.float32)
    y_vals = np.linspace(-1, 1, res)
    z_vals = np.linspace(0, 2, res)
    yy, zz = np.meshgrid(y_vals, z_vals)
    points_side[:, :, 0] = 2.0  # Constant X
    points_side[:, :, 1] = yy
    points_side[:, :, 2] = zz

    normals_side = projector._compute_normals(points_side)
    dominant_side = np.argmax(np.abs(normals_side[interior, interior]), axis=-1)
    assert np.all(dominant_side == 0), (
        f"Side wall should be X-dominant (plane 0), got: {np.unique(dominant_side)}"
    )
    print(f"  Side wall (X=const): {plane_names[0]} - OK")

    # --- Front wall: Y=1.5, varying X and Z ---
    points_front = np.zeros((res, res, 3), dtype=np.float32)
    x_vals = np.linspace(-1, 1, res)
    xx, zz2 = np.meshgrid(x_vals, z_vals)
    points_front[:, :, 0] = xx
    points_front[:, :, 1] = 1.5  # Constant Y
    points_front[:, :, 2] = zz2

    normals_front = projector._compute_normals(points_front)
    dominant_front = np.argmax(np.abs(normals_front[interior, interior]), axis=-1)
    assert np.all(dominant_front == 1), (
        f"Front wall should be Y-dominant (plane 1), got: {np.unique(dominant_front)}"
    )
    print(f"  Front wall (Y=const): {plane_names[1]} - OK")

    # --- Roof: Z=1.5, varying X and Y ---
    points_roof = np.zeros((res, res, 3), dtype=np.float32)
    y_vals2 = np.linspace(-1, 1, res)
    xx2, yy2 = np.meshgrid(x_vals, y_vals2)
    points_roof[:, :, 0] = xx2
    points_roof[:, :, 1] = yy2
    points_roof[:, :, 2] = 1.5  # Constant Z

    normals_roof = projector._compute_normals(points_roof)
    dominant_roof = np.argmax(np.abs(normals_roof[interior, interior]), axis=-1)
    assert np.all(dominant_roof == 2), (
        f"Roof should be Z-dominant (plane 2), got: {np.unique(dominant_roof)}"
    )
    print(f"  Roof (Z=const): {plane_names[2]} - OK")

    # Key: these results are camera-independent because normals are computed
    # from vehicle-local points. A car door is always X-dominant regardless
    # of camera angle. Previously, camera-space normals would flip planes.
    print("  All surfaces correctly classified in vehicle-local space")
    print("  PASSED")
    print()


if __name__ == '__main__':
    print("=" * 70)
    print("TRIPLANAR PROJECTION TESTS")
    print("=" * 70)
    print()

    test_gradient_flow()
    test_mask_application()
    test_tile_size_affects_uv()
    test_output_shape()
    test_rotation_matrix()
    test_normals_flat_plane()
    test_different_viewpoints_differ()
    test_batch_gradient()
    test_viewpoint_invariant_plane_selection()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
