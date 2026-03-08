#!/usr/bin/env python3
"""
DTA-Style Repeated Texture Projection Function (PyTorch)

Implements viewpoint-consistent texture projection following the DTA paper
(CVPR 2022, Eq. 5, Fig. 4). Transforms a small texture pattern (e.g., 8x8)
into full-resolution (1024x1024) tiled projections that shift, scale, and warp
based on camera viewpoint parameters.

Key Operations (Eq. 5: eta_adv_p = M_3DRot * M_scale * M_shift * eta_adv_b):
    M_shift: Horizontal shift based on yaw angle (wraps around vehicle)
    M_scale: Scale based on camera distance (closer = larger tiles)
    M_3DRot: Perspective foreshortening based on viewing angle

All operations are differentiable via F.grid_sample, enabling gradient flow
from the projected texture back to the optimizable pattern.

Usage:
    from attack.texture_projection import RepeatedTextureProjection

    projector = RepeatedTextureProjection(tile_count=8)
    # Single viewpoint
    projected = projector.project(texture, yaw=60, pitch=-15, distance=8)
    # Batch of viewpoints
    projected = projector.project_batch(texture, viewpoints)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


class RepeatedTextureProjection:
    """
    DTA-style repeated texture projection with differentiable grid sampling.

    Projects a small optimizable texture into a full-resolution tiled pattern
    that is viewpoint-consistent: different camera angles produce shifted/scaled
    versions of the same pattern, as if printed on the vehicle surface.
    """

    def __init__(self, tile_count=8, reference_distance=8.0, full_resolution=1024,
                 boxiness=0.4, min_foreshorten=0.2):
        """
        Args:
            tile_count: Number of texture repetitions per axis (8 = 8x8 tiling)
            reference_distance: Distance at which scale factor = 1.0
            full_resolution: Output resolution (square)
            boxiness: Non-linearity of yaw-to-shift mapping (0=cylinder, ~0.4=boxy car).
                Slows shift at face centers (0°,90°,...) and speeds it at corners (45°,135°,...).
            min_foreshorten: Minimum foreshortening factor at corner angles (0.5=soft, 0.2=aggressive)
        """
        self.tile_count = tile_count
        self.reference_distance = reference_distance
        self.full_resolution = full_resolution
        self.boxiness = boxiness
        self.min_foreshorten = min_foreshorten

    def project(self, texture, yaw, pitch, distance, sample_mode='nearest'):
        """
        Project texture for a single viewpoint.

        Applies the DTA transformation sequence:
            1. Scale grid by tile_count (creates tiling)
            2. M_shift: Horizontal shift proportional to yaw
            3. M_scale: Scale proportional to reference_distance / distance
            4. M_3DRot: Horizontal compression via cos(yaw) for foreshortening
            5. Wrap mode: differentiable modular arithmetic for seamless tiling
            6. F.grid_sample (nearest: differentiable w.r.t. texture, matches Phase 1)

        Args:
            texture: [3, H_tex, W_tex] or [1, 3, H_tex, W_tex] float32 [0, 1]
            yaw: Camera yaw angle in degrees (0-360)
            pitch: Camera pitch angle in degrees (negative = looking down)
            distance: Camera distance from vehicle in meters
            sample_mode: Interpolation mode ('nearest' or 'bilinear')

        Returns:
            projected: [3, full_res, full_res] float32 [0, 1]
        """
        squeeze = False
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)  # [1, 3, H, W]
            squeeze = True

        device = texture.device
        res = self.full_resolution

        # Create base identity grid in [-1, 1]
        # grid[y, x] = (x_coord, y_coord) for sampling
        grid = self._make_base_grid(res, device)  # [1, H, W, 2]

        # 1. Scale by tile_count: maps [-1,1] to [-tile_count, tile_count]
        #    This means the texture is sampled tile_count times per axis
        grid = grid * self.tile_count

        # 2. M_shift: horizontal shift based on yaw (non-linear for boxy shapes)
        #    x_shift = (θ - (boxiness/4)·sin(4θ)) / π
        #    This slows the shift at face centers (0°, 90°, ...) and speeds it
        #    at corners (45°, 135°, ...), matching a box-like vehicle silhouette.
        #    boxiness=0 recovers the original linear mapping.
        yaw_rad_full = math.radians(yaw)
        yaw_shift = (yaw_rad_full - (self.boxiness / 4) * math.sin(4 * yaw_rad_full)) / math.pi
        grid[..., 0] = grid[..., 0] + yaw_shift

        # 3. M_scale: zoom based on distance
        #    Closer camera = larger apparent tiles (fewer repetitions visible)
        #    Farther camera = smaller apparent tiles (more repetitions visible)
        #    scale < 1 when closer (shrinks grid = fewer tiles = bigger),
        #    scale > 1 when farther (expands grid = more tiles = smaller)
        scale = distance / max(self.reference_distance, 1e-6)
        grid = grid * scale

        # 4. M_3DRot: perspective foreshortening
        #    Approximate the car surface as a plane; when viewed from the side,
        #    the horizontal extent compresses by cos(effective_angle).
        #    We use yaw mod 90 to get the angle relative to the nearest
        #    face-on direction (0°, 90°, 180°, 270°).
        yaw_rad = math.radians(yaw % 360)
        # Foreshortening factor: how much the surface is angled away from camera
        # At 0° (front-on), cos=1 (no compression). At 45°, cos≈0.7 (compressed).
        # Use a softened version to avoid extreme compression near 90°.
        angle_from_face = yaw_rad % (math.pi / 2)  # 0 to 90° range
        foreshorten = math.cos(angle_from_face)
        # Blend toward 1.0 using tunable min_foreshorten
        foreshorten = self.min_foreshorten + (1 - self.min_foreshorten) * foreshorten
        grid[..., 0] = grid[..., 0] * foreshorten

        # Vertical foreshortening from pitch
        pitch_rad = math.radians(abs(pitch))
        pitch_foreshorten = math.cos(pitch_rad)
        pitch_foreshorten = self.min_foreshorten + (1 - self.min_foreshorten) * pitch_foreshorten
        grid[..., 1] = grid[..., 1] * pitch_foreshorten

        # 5. Wrap mode: differentiable modular arithmetic
        #    Map grid coordinates to [-1, 1] with seamless wrapping
        grid = self._differentiable_wrap(grid)

        # 6. Sample texture using grid
        projected = F.grid_sample(
            texture, grid,
            mode=sample_mode,
            padding_mode='border',
            align_corners=True
        )  # [1, 3, res, res]

        if squeeze:
            projected = projected.squeeze(0)  # [3, res, res]

        return projected

    def project_batch(self, texture, viewpoints):
        """
        Project texture for multiple viewpoints.

        Args:
            texture: [3, H_tex, W_tex] float32 [0, 1] (single optimizable pattern)
            viewpoints: List of dicts with 'yaw', 'pitch', 'distance' keys

        Returns:
            projected: [B, 3, full_res, full_res] float32 [0, 1]
        """
        projections = []
        for vp in viewpoints:
            proj = self.project(
                texture,
                yaw=vp['yaw'],
                pitch=vp['pitch'],
                distance=vp['distance']
            )
            projections.append(proj)

        return torch.stack(projections, dim=0)  # [B, 3, res, res]

    def project_custom(self, texture, yaw, pitch, distance, sample_mode='nearest'):
        """
        Project texture with corrected distance scaling.

        Like project(), but uses inverse distance scaling so that
        texture features maintain consistent physical size on the
        vehicle bodywork regardless of camera distance.

        DTA uses scale = d_ref / d (closer = bigger tiles, wrong).
        This uses scale = d / d_ref (closer = finer tiles, correct).

        Args:
            texture: [3, H_tex, W_tex] or [1, 3, H_tex, W_tex] float32 [0, 1]
            yaw: Camera yaw angle in degrees (0-360)
            pitch: Camera pitch angle in degrees (negative = looking down)
            distance: Camera distance from vehicle in meters
            sample_mode: Interpolation mode for grid_sample ('bilinear' or 'nearest')

        Returns:
            projected: [3, full_res, full_res] float32 [0, 1]
        """
        squeeze = False
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)  # [1, 3, H, W]
            squeeze = True

        device = texture.device
        res = self.full_resolution

        # Create base identity grid in [-1, 1]
        grid = self._make_base_grid(res, device)  # [1, H, W, 2]

        # 1. Scale by tile_count: maps [-1,1] to [-tile_count, tile_count]
        grid = grid * self.tile_count

        # 2. M_shift: horizontal shift based on yaw (non-linear for boxy shapes)
        yaw_rad_full = math.radians(yaw)
        yaw_shift = (yaw_rad_full - (self.boxiness / 4) * math.sin(4 * yaw_rad_full)) / math.pi
        grid[..., 0] = grid[..., 0] + yaw_shift

        # 3. M_scale: INVERTED — closer camera = finer tiles
        scale = distance / max(self.reference_distance, 1e-6)
        grid = grid * scale

        # 4. M_3DRot: perspective foreshortening (identical to project())
        yaw_rad = math.radians(yaw % 360)
        angle_from_face = yaw_rad % (math.pi / 2)
        foreshorten = math.cos(angle_from_face)
        foreshorten = self.min_foreshorten + (1 - self.min_foreshorten) * foreshorten
        grid[..., 0] = grid[..., 0] * foreshorten

        # Vertical foreshortening from pitch
        pitch_rad = math.radians(abs(pitch))
        pitch_foreshorten = math.cos(pitch_rad)
        pitch_foreshorten = self.min_foreshorten + (1 - self.min_foreshorten) * pitch_foreshorten
        grid[..., 1] = grid[..., 1] * pitch_foreshorten

        # 5. Wrap mode: differentiable modular arithmetic
        grid = self._differentiable_wrap(grid)

        # 6. Sample texture using grid
        projected = F.grid_sample(
            texture, grid,
            mode=sample_mode,
            padding_mode='border',
            align_corners=True
        )  # [1, 3, res, res]

        if squeeze:
            projected = projected.squeeze(0)  # [3, res, res]

        return projected

    def project_custom_batch(self, texture, viewpoints):
        """
        Project texture for multiple viewpoints using corrected distance scaling.

        Args:
            texture: [3, H_tex, W_tex] float32 [0, 1] (single optimizable pattern)
            viewpoints: List of dicts with 'yaw', 'pitch', 'distance' keys

        Returns:
            projected: [B, 3, full_res, full_res] float32 [0, 1]
        """
        projections = []
        for vp in viewpoints:
            proj = self.project_custom(
                texture,
                yaw=vp['yaw'],
                pitch=vp['pitch'],
                distance=vp['distance']
            )
            projections.append(proj)

        return torch.stack(projections, dim=0)  # [B, 3, res, res]

    def project_manual(self, texture, x_shift=0.0, y_shift=0.0, scale=1.0, scale_ratio=1.0):
        """
        Project texture with explicit manual controls.

        Args:
            texture: [3, H, W] or [1, 3, H, W] float32 [0, 1]
            x_shift: Horizontal translation in grid units (2.0 = one full tile period)
            y_shift: Vertical translation in grid units
            scale: Zoom factor (>1 = larger tiles, <1 = smaller tiles)
            scale_ratio: X/Y aspect ratio (>1 = wider tiles, <1 = taller tiles)

        Returns: [3, full_res, full_res] float32 [0, 1]
        """
        # Handle 3D/4D input
        squeeze = False
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)
            squeeze = True

        grid = self._make_base_grid(self.full_resolution, texture.device)

        # 1. Tiling
        grid = grid * self.tile_count

        # 2. Scale (divide: larger scale = bigger tiles)
        grid = grid / max(scale, 1e-6)

        # 3. Scale ratio (stretch X axis)
        grid[..., 0] = grid[..., 0] / max(scale_ratio, 1e-6)

        # 4. Shift
        grid[..., 0] = grid[..., 0] + x_shift
        grid[..., 1] = grid[..., 1] + y_shift

        # 5. Wrap + sample
        grid = self._differentiable_wrap(grid)
        projected = F.grid_sample(texture, grid, mode='nearest',
                                  padding_mode='border', align_corners=True)

        if squeeze:
            projected = projected.squeeze(0)
        return projected

    def _make_base_grid(self, resolution, device):
        """
        Create base sampling grid in [-1, 1].

        Returns:
            grid: [1, H, W, 2] where grid[0, y, x] = (x_coord, y_coord)
        """
        coords = torch.linspace(-1, 1, resolution, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        return grid.unsqueeze(0)  # [1, H, W, 2]

    def _differentiable_wrap(self, grid):
        """
        Differentiable wrap-around for seamless texture tiling.

        Maps arbitrary grid coordinates back to [-1, 1] using modular arithmetic.
        Uses torch.remainder which is differentiable.

        Args:
            grid: [B, H, W, 2] arbitrary coordinates

        Returns:
            grid: [B, H, W, 2] coordinates in [-1, 1]
        """
        # Map from [-inf, inf] to [0, 2] then to [-1, 1]
        grid = torch.remainder(grid + 1.0, 2.0) - 1.0
        return grid


class TriplanarProjection:
    """
    Projects texture onto car surface using depth-based triplanar UV mapping.

    Uses CARLA depth data to compute actual 3D positions of car pixels,
    then maps to texture coordinates using triplanar projection.
    Differentiable w.r.t. texture via F.grid_sample.
    """

    def __init__(self, tile_size=0.5, resolution=1024, fov=90.0, blend_sharpness=4.0):
        """
        Args:
            tile_size: Physical size of one texture tile in meters (default 0.5m)
            resolution: Image resolution (square)
            fov: Camera field of view in degrees (CARLA default: 90)
            blend_sharpness: Exponent for triplanar blending (higher = sharper face transitions)
        """
        self.tile_size = tile_size
        self.resolution = resolution
        self.fov = fov
        self.blend_sharpness = blend_sharpness

        # Pre-compute camera intrinsics
        self.focal = resolution / (2.0 * math.tan(math.radians(fov / 2.0)))
        self.cx = resolution / 2.0
        self.cy = resolution / 2.0

    def project(self, texture, depth, mask, camera_transform, vehicle_transform):
        """
        Project texture onto car surface using depth-based triplanar mapping.

        Args:
            texture: [3, H_tex, W_tex] or [1, 3, H_tex, W_tex] float32 [0, 1]
            depth: numpy [H, W] float32, depth in meters from CARLA
            mask: numpy [H, W] float32, binary car mask (1.0 = car)
            camera_transform: carla.Transform of camera
            vehicle_transform: carla.Transform of vehicle

        Returns:
            projected: [3, H, W] float32 [0, 1], texture projected onto car pixels
        """
        squeeze = False
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)  # [1, 3, H, W]
            squeeze = True

        device = texture.device
        H, W = depth.shape

        # Step 1: Back-project depth to 3D camera space
        # Build pixel coordinate grids
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)  # [H, W] each

        # Camera-space 3D points: P_cam = [(u-cx)*D/f, (v-cy)*D/f, D]
        # CARLA camera: X=right, Y=down, Z=forward
        x_cam = (u_grid - self.cx) * depth / self.focal
        y_cam = (v_grid - self.cy) * depth / self.focal
        z_cam = depth

        # Stack to [H, W, 3]
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Step 2: Transform camera space -> world space -> car-local space
        points_local = self._cam_to_vehicle_local(
            points_cam, camera_transform, vehicle_transform
        )  # [H, W, 3]

        # Step 3: Compute surface normals from depth gradients
        normals = self._compute_normals(points_local)  # [H, W, 3]

        # Step 4: Build single grid via hard argmax selection of dominant plane
        grid = self._triplanar_grid_hard(points_local, normals, device)  # [1, H, W, 2]

        # Step 5: Sample texture once
        projected = F.grid_sample(
            texture, grid,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        )  # [1, 3, H, W]

        if squeeze:
            projected = projected.squeeze(0)  # [3, H, W]

        # Step 6: Apply car mask (zero out non-car pixels)
        mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)  # [1, H, W]
        projected = projected * mask_t

        return projected

    def get_debug_plane_map(self, depth, mask, camera_transform, vehicle_transform):
        """
        Return [H, W, 3] uint8 RGB image color-coding the dominant projection plane.

        Colors:
            Red   = side  (X-dominant normal, YZ projection — doors, fenders)
            Green = front (Y-dominant normal, XZ projection — bumpers)
            Blue  = top   (Z-dominant normal, XY projection — hood, roof)

        Args:
            depth: numpy [H, W] float32, depth in meters from CARLA
            mask: numpy [H, W] float32, binary car mask (1.0 = car)
            camera_transform: carla.Transform of camera
            vehicle_transform: carla.Transform of vehicle

        Returns:
            plane_map: [H, W, 3] uint8 RGB image
        """
        H, W = depth.shape

        # Back-project depth to 3D camera space
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)

        x_cam = (u_grid - self.cx) * depth / self.focal
        y_cam = (v_grid - self.cy) * depth / self.focal
        z_cam = depth
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to vehicle-local space
        points_local = self._cam_to_vehicle_local(
            points_cam, camera_transform, vehicle_transform
        )

        # Compute normals from vehicle-local points
        normals = self._compute_normals(points_local)

        # Determine dominant axis per pixel
        abs_normals = np.abs(normals)
        dominant = np.argmax(abs_normals, axis=-1)  # [H, W], values 0/1/2

        # Build color map: 0=Red(side), 1=Green(front), 2=Blue(top)
        plane_map = np.zeros((H, W, 3), dtype=np.uint8)
        plane_map[dominant == 0] = [255, 0, 0]    # Red = side (X-dominant)
        plane_map[dominant == 1] = [0, 255, 0]    # Green = front (Y-dominant)
        plane_map[dominant == 2] = [0, 0, 255]    # Blue = top (Z-dominant)

        # Apply car mask
        mask_bool = mask > 0.5
        plane_map[~mask_bool] = 0

        return plane_map

    def _cam_to_vehicle_local(self, points_cam, camera_transform, vehicle_transform):
        """
        Transform points from camera space to vehicle-local coordinates.

        Args:
            points_cam: [H, W, 3] float32, camera-space points (X=right, Y=down, Z=forward)
            camera_transform: carla.Transform of camera
            vehicle_transform: carla.Transform of vehicle

        Returns:
            points_local: [H, W, 3] float32, vehicle-local coordinates
        """
        H, W, _ = points_cam.shape
        pts_flat = points_cam.reshape(-1, 3)  # [N, 3]

        # Camera rotation (CARLA convention: pitch, yaw, roll in degrees)
        cam_rot = camera_transform.rotation
        cam_loc = camera_transform.location

        # Build camera-to-world rotation matrix
        # CARLA uses left-handed coordinate system
        # Camera space: X=right, Y=down, Z=forward
        # World space: X=forward, Y=right, Z=up
        R_cam = self._rotation_matrix(cam_rot.pitch, cam_rot.yaw, cam_rot.roll)

        # Camera-to-world transform for each point:
        # P_world = R_cam @ P_cam_reordered + T_cam
        # First reorder: camera (right, down, forward) -> UE4 (forward, right, up)
        # UE4/CARLA: X=forward, Y=right, Z=up
        # Camera: X=right, Y=down, Z=forward
        # Mapping: cam_X -> world_Y, cam_Y -> world_-Z, cam_Z -> world_X
        pts_ue4 = np.stack([
            pts_flat[:, 2],   # cam Z (forward) -> world X (forward)
            pts_flat[:, 0],   # cam X (right) -> world Y (right)
            -pts_flat[:, 1],  # cam -Y (up) -> world Z (up)
        ], axis=-1)  # [N, 3]

        # Apply camera rotation
        pts_world = (R_cam @ pts_ue4.T).T  # [N, 3]

        # Add camera translation
        pts_world[:, 0] += cam_loc.x
        pts_world[:, 1] += cam_loc.y
        pts_world[:, 2] += cam_loc.z

        # Now transform world -> vehicle local
        veh_rot = vehicle_transform.rotation
        veh_loc = vehicle_transform.location

        # Subtract vehicle position
        pts_world[:, 0] -= veh_loc.x
        pts_world[:, 1] -= veh_loc.y
        pts_world[:, 2] -= veh_loc.z

        # Apply inverse vehicle rotation
        R_veh = self._rotation_matrix(veh_rot.pitch, veh_rot.yaw, veh_rot.roll)
        R_veh_inv = R_veh.T  # Orthogonal matrix, inverse = transpose
        pts_local = (R_veh_inv @ pts_world.T).T  # [N, 3]

        return pts_local.reshape(H, W, 3)

    def _rotation_matrix(self, pitch, yaw, roll):
        """
        Build 3x3 rotation matrix from CARLA Euler angles (degrees).

        CARLA uses Unreal Engine convention:
        - Rotation order: Yaw (Z-axis), Pitch (Y-axis), Roll (X-axis)
        - Left-handed coordinate system

        Returns:
            R: [3, 3] numpy rotation matrix
        """
        p = math.radians(pitch)
        y = math.radians(yaw)
        r = math.radians(roll)

        # Individual rotation matrices (UE4 convention, left-handed)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        cr, sr = math.cos(r), math.sin(r)

        # Combined rotation: R = Ryaw * Rpitch * Rroll
        R = np.array([
            [cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ],
        ], dtype=np.float32)

        return R

    def _compute_normals(self, points):
        """
        Compute surface normals from 3D points using finite differences.

        Args:
            points: [H, W, 3] float32 (any coordinate space)

        Returns:
            normals: [H, W, 3] float32, unit normals
        """
        H, W, _ = points.shape

        # Finite differences (shifted by 1 pixel)
        dPdu = np.zeros_like(points)
        dPdv = np.zeros_like(points)

        dPdu[:, :-1, :] = points[:, 1:, :] - points[:, :-1, :]
        dPdu[:, -1, :] = dPdu[:, -2, :]  # Replicate last column

        dPdv[:-1, :, :] = points[1:, :, :] - points[:-1, :, :]
        dPdv[-1, :, :] = dPdv[-2, :, :]  # Replicate last row

        # Cross product for normal
        normals = np.cross(dPdu, dPdv)  # [H, W, 3]

        # Normalize
        norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        norm_len = np.maximum(norm_len, 1e-8)
        normals = normals / norm_len

        return normals

    def _triplanar_grid_hard(self, points_local, normals, device):
        """
        Build a single sampling grid using hard argmax plane selection.

        For each pixel, the dominant surface normal axis determines which
        projection plane to use. No blending — each pixel gets UV from
        exactly one plane, preserving clean tile structure everywhere.

        Planes:
            |nx| dominant (side faces like doors): UV = (py, pz)
            |ny| dominant (front/back like bumpers): UV = (px, pz)
            |nz| dominant (top/bottom like hood/roof): UV = (px, py)

        Args:
            points_local: [H, W, 3] float32, vehicle-local 3D positions
            normals: [H, W, 3] float32, surface normals
            device: torch device

        Returns:
            grid: [1, H, W, 2] torch tensor, UV coordinates in [-1, 1]
        """
        H, W, _ = points_local.shape

        px = points_local[:, :, 0]  # forward
        py = points_local[:, :, 1]  # right
        pz = points_local[:, :, 2]  # up

        # Determine dominant axis per pixel
        abs_normals = np.abs(normals)  # [H, W, 3]
        dominant = np.argmax(abs_normals, axis=-1)  # [H, W], values 0/1/2

        # Build UV grid by selecting from the appropriate plane per pixel
        uv = np.zeros((H, W, 2), dtype=np.float32)

        side = dominant == 0   # X-dominant → YZ plane
        front = dominant == 1  # Y-dominant → XZ plane
        top = dominant == 2    # Z-dominant → XY plane

        uv[side, 0] = py[side] / self.tile_size
        uv[side, 1] = pz[side] / self.tile_size

        uv[front, 0] = px[front] / self.tile_size
        uv[front, 1] = pz[front] / self.tile_size

        uv[top, 0] = px[top] / self.tile_size
        uv[top, 1] = py[top] / self.tile_size

        # Wrap to [-1, 1] for seamless tiling
        uv = np.remainder(uv + 1.0, 2.0) - 1.0

        grid = torch.from_numpy(uv).unsqueeze(0).to(device)  # [1, H, W, 2]
        return grid
