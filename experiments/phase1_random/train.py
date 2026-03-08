#!/usr/bin/env python3
"""
Phase 2: Robust EOT Adversarial Texture Optimization

Extends Phase 1 by expanding the EOT distribution to include:
- Multiple pitch angles (5, 10, 15 degrees)
- Multiple distances (6, 8, 10, 12 m)
- 6 weather presets (no heavy rain, no night)
- Sequential spawn point cycling (50 points)
- Per-viewpoint texture projection via RepeatedTextureProjection

Pipeline:
    CARLA (reference images + masks across varied conditions)
    -> Per-viewpoint DTA-style texture projection (16x16 -> 1024x1024)
    -> UNet3 Neural Renderer (differentiable)
    -> EfficientDet-D0 (pre-NMS with grad)
    -> Attack Loss (minimize car confidence)
    -> Adam optimizer (update 16x16 texture)

Requirements:
    - CARLA server running (launched by run.sh)
    - Trained U-Net3 renderer: models/unet3/trained/best_model.pt
    - conda environment: camo

Usage:
    # Edit run.sh to launch this script, then:
    ./run.sh
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import cv2
import numpy as np
import time
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from scripts.CarlaHandler import CarlaHandler
from scripts.texture_applicator_pytorch import TextureApplicatorPyTorch
from attack.detector_pytorch import EfficientDetPyTorch
from attack.texture_projection import RepeatedTextureProjection
from attack.logger import CSVLogger


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    'learning_rate': 0.01,
    'num_iterations': 2000,
    'checkpoint_every': 100,
    'log_every': 10,
    'coarse_size': 16,
    'tile_count': 8,
    'full_resolution': 1024,
    'detector_input_size': 512,
    'clip_grad_norm': 1.0,
    'views_per_batch': 4,
    'views_per_iteration': 12,
    'views_to_cache': 24,
    'spawn_weather_interval': 10,
    'yaws': list(range(0, 360, 30)),
    'pitches': [5, 10, 15],
    'distances': [6, 8, 10, 12],
    'reference_distance': 8.0,
    'output_dir': 'experiments/phase2_robust_eot/',
    'town': 'Town03',
}


# ─── Weather Presets ────────────────────────────────────────────────────────

WEATHER_PRESETS = [
    {'name': 'clear_noon',      'cloudiness': 0,  'sun_altitude': 60, 'sun_azimuth': 0,   'fog_density': 0},
    {'name': 'clear_afternoon', 'cloudiness': 10, 'sun_altitude': 40, 'sun_azimuth': 120, 'fog_density': 0},
    {'name': 'partly_cloudy',   'cloudiness': 40, 'sun_altitude': 50, 'sun_azimuth': 200, 'fog_density': 0},
    {'name': 'overcast',        'cloudiness': 80, 'sun_altitude': 30, 'sun_azimuth': 90,  'fog_density': 0},
    {'name': 'golden_hour',     'cloudiness': 15, 'sun_altitude': 8,  'sun_azimuth': 270, 'fog_density': 0},
    {'name': 'hazy',            'cloudiness': 30, 'sun_altitude': 45, 'sun_azimuth': 150, 'fog_density': 25},
]


# ─── Helpers ────────────────────────────────────────────────────────────────

def create_viewpoint_pool(yaws, pitches, distances):
    """Create full viewpoint pool from yaw x pitch x distance combinations."""
    pool = []
    for yaw in yaws:
        for pitch in pitches:
            for dist in distances:
                pool.append({'yaw': yaw, 'pitch': pitch, 'distance': dist})
    return pool


def visualize_texture(texture_np, save_path):
    """Save texture as PNG for visual inspection."""
    texture_uint8 = (texture_np * 255).astype(np.uint8)
    texture_bgr = cv2.cvtColor(texture_uint8, cv2.COLOR_RGB2BGR)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), texture_bgr)


# ─── Trainer ────────────────────────────────────────────────────────────────

class RobustEOTTrainer:
    """
    Phase 2 EOT trainer with expanded distribution over pitch, distance,
    weather, and spawn points. Uses per-viewpoint texture projection.
    """

    def __init__(self, carla_handler, detector, renderer, projector, config):
        self.carla = carla_handler
        self.detector = detector
        self.renderer = renderer
        self.projector = projector
        self.config = config
        self.device = next(detector.model.parameters()).device
        print(f"RobustEOTTrainer using device: {self.device}")

        if self.carla.vehicle is None:
            raise ValueError("CarlaHandler must have a vehicle spawned")

        # Use all spawn points the map provides
        self.num_spawn_points = len(self.carla.spawn_points)
        print(f"  Spawn points: {self.num_spawn_points}")

        # Build viewpoint pool (12 yaws x 3 pitches x 4 distances = 144)
        self.viewpoint_pool = create_viewpoint_pool(
            config['yaws'], config['pitches'], config['distances']
        )
        print(f"  Viewpoint pool: {len(self.viewpoint_pool)} combinations")

        # Configure camera (initial setup, will be overridden per viewpoint)
        self.carla.update_view('3d')
        self.carla.update_pitch(15)
        self.carla.update_distance(8)
        self.carla.update_yaw(-1)  # dummy so first real update triggers move
        self.carla.world.tick(60)
        time.sleep(1.0)

        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.viz_dir = self.output_dir / 'visualizations'
        self.final_dir = self.output_dir / 'final'

        self.checkpoint_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        self.final_dir.mkdir(exist_ok=True)

        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # State
        self.spawn_index = 0
        self.current_weather = None
        self.cached_viewpoints = []
        self.cached_refs = None
        self.cached_masks = None

        print(f"RobustEOTTrainer initialized")
        print(f"  Output: {self.output_dir}")

    def set_weather(self, preset):
        """Apply a weather preset to the CARLA world."""
        self.carla.weather.cloudiness = preset['cloudiness']
        self.carla.weather.sun_altitude_angle = preset['sun_altitude']
        self.carla.weather.sun_azimuth_angle = preset['sun_azimuth']
        self.carla.weather.fog_density = preset['fog_density']
        self.carla.weather.precipitation = 0.0
        self.carla.weather.precipitation_deposits = 0.0
        self.carla.weather.wind_intensity = 0.0
        self.carla.weather.fog_distance = 0.0
        self.carla.world.set_weather(self.carla.weather)
        self.current_weather = preset

    def _validate_spawn_point(self):
        """Check if current spawn point has a visible car by testing a few viewpoints.

        Tests 4 cardinal yaw angles — if any has a valid mask, the spawn point is usable.

        Returns:
            True if car mask has enough pixels at any test viewpoint.
        """
        for test_yaw in [0, 90, 180, 270]:
            self.carla.spectator_yaw = test_yaw
            self.carla.spectator_pitch = 10
            self.carla.spectator_distance = 8
            self.carla._update_spectator()
            self.carla.world.tick(10)
            time.sleep(0.1)

            mask = self.carla.get_car_segmentation_mask()
            if (mask > 0.5).sum() > 1000:
                return True

        return False

    def _refresh_spawn_weather_cache(self):
        """Change spawn point + weather, capture reference images for cached viewpoints."""
        # Try spawn points until we find one with a visible car
        max_attempts = self.num_spawn_points
        for attempt in range(max_attempts):
            self.carla.change_spawn_point(self.spawn_index)
            current_spawn = self.spawn_index
            self.spawn_index = (self.spawn_index + 1) % self.num_spawn_points
            self.carla.world.tick(30)
            time.sleep(0.5)

            if self._validate_spawn_point():
                break

            print(f"  Spawn {current_spawn}: car not visible, skipping...")
        else:
            print(f"  WARNING: No valid spawn point found after {max_attempts} attempts, "
                  f"using last one")

        # Random weather
        preset = random.choice(WEATHER_PRESETS)
        self.set_weather(preset)

        # Heavy settle after spawn + weather change
        self.carla.world.tick(60)
        time.sleep(1.0)

        # Sample random viewpoints from pool
        views_to_cache = self.config['views_to_cache']
        self.cached_viewpoints = random.sample(
            self.viewpoint_pool,
            min(views_to_cache, len(self.viewpoint_pool))
        )

        # Capture references and masks, filtering out views with no car visible
        self.cached_refs, self.cached_masks, self.cached_viewpoints = \
            self._capture_viewpoints_filtered(self.cached_viewpoints)

        print(f"  Cache refreshed: spawn={current_spawn}, "
              f"weather={preset['name']}, views={len(self.cached_viewpoints)}")

    def _capture_viewpoints(self, viewpoints):
        """
        Capture reference images and masks for given viewpoints.

        Args:
            viewpoints: List of dicts with 'yaw', 'pitch', 'distance' keys

        Returns:
            x_ref_batch: torch.Tensor [N, 3, H, W] float32 [0, 1]
            mask_batch: torch.Tensor [N, 1, H, W] float32 [0, 1]
        """
        x_refs = []
        masks = []

        for vp in viewpoints:
            # Set viewpoint attributes directly, call _update_spectator once
            self.carla.spectator_yaw = vp['yaw']
            self.carla.spectator_pitch = vp['pitch']
            self.carla.spectator_distance = vp['distance']
            self.carla._update_spectator()

            # Settle after camera move — all ticks BEFORE capturing
            self.carla.world.tick(20)
            time.sleep(0.2)

            # Capture image and mask back-to-back (no ticks between!)
            img = self.carla.get_image()
            car_mask = self.carla.get_car_segmentation_mask()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0

            x_refs.append(img_float)
            masks.append(car_mask)

        # Stack and convert to tensors (NHWC -> NCHW)
        x_ref_np = np.stack(x_refs, axis=0)                       # [N, H, W, 3]
        x_ref_np = np.transpose(x_ref_np, (0, 3, 1, 2))          # [N, 3, H, W]
        x_ref_batch = torch.from_numpy(x_ref_np).float().to(self.device)

        mask_np = np.stack(masks, axis=0)[:, :, :, np.newaxis]    # [N, H, W, 1]
        mask_np = np.transpose(mask_np, (0, 3, 1, 2))             # [N, 1, H, W]
        mask_batch = torch.from_numpy(mask_np).float().to(self.device)

        return x_ref_batch, mask_batch

    def _capture_viewpoints_filtered(self, viewpoints, min_mask_pixels=1000):
        """Capture viewpoints, dropping any where the car mask is too small.

        Returns:
            x_ref_batch: torch.Tensor [M, 3, H, W]
            mask_batch: torch.Tensor [M, 1, H, W]
            valid_viewpoints: List of M viewpoint dicts (subset of input)
        """
        x_refs = []
        masks = []
        valid_viewpoints = []

        for vp in viewpoints:
            self.carla.spectator_yaw = vp['yaw']
            self.carla.spectator_pitch = vp['pitch']
            self.carla.spectator_distance = vp['distance']
            self.carla._update_spectator()

            # Settle after camera move — all ticks BEFORE capturing
            self.carla.world.tick(20)
            time.sleep(0.2)

            # Capture image and mask back-to-back (no ticks between!)
            img = self.carla.get_image()
            car_mask = self.carla.get_car_segmentation_mask()

            mask_pixels = (car_mask > 0.5).sum()
            if mask_pixels < min_mask_pixels:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0

            x_refs.append(img_float)
            masks.append(car_mask)
            valid_viewpoints.append(vp)

        if len(x_refs) == 0:
            raise RuntimeError("No valid viewpoints found — car not visible from any angle")

        x_ref_np = np.stack(x_refs, axis=0)
        x_ref_np = np.transpose(x_ref_np, (0, 3, 1, 2))
        x_ref_batch = torch.from_numpy(x_ref_np).float().to(self.device)

        mask_np = np.stack(masks, axis=0)[:, :, :, np.newaxis]
        mask_np = np.transpose(mask_np, (0, 3, 1, 2))
        mask_batch = torch.from_numpy(mask_np).float().to(self.device)

        return x_ref_batch, mask_batch, valid_viewpoints

    def _forward_pass(self, texture, viewpoints, x_ref_batch, mask_batch,
                      return_intermediates=False):
        """
        Forward pass with per-viewpoint texture projection and gradient accumulation.

        Processes viewpoints in mini-batches, calling .backward() per mini-batch
        so computation graphs are freed immediately. Gradients accumulate on texture.

        Args:
            texture: [3, coarse_size, coarse_size] (requires_grad=True)
            viewpoints: List of viewpoint dicts for projection
            x_ref_batch: [N, 3, H, W] reference images
            mask_batch: [N, 1, H, W] car masks
            return_intermediates: If True, return detached CPU copies of intermediates

        Returns:
            loss_value: float (detached, for logging)
            metrics: Dict with max/mean confidence and per_view_conf
            intermediates: (only if return_intermediates=True) Dict of detached CPU tensors
        """
        detector_size = self.config['detector_input_size']
        mini_batch = self.config['views_per_batch']
        num_views = len(viewpoints)

        all_max_conf = []
        total_loss = 0.0
        int_tex_projected = [] if return_intermediates else None
        int_rendered = [] if return_intermediates else None
        int_resized = [] if return_intermediates else None

        for start in range(0, num_views, mini_batch):
            end = min(start + mini_batch, num_views)
            bs = end - start

            x_ref_mb = x_ref_batch[start:end]
            mask_mb = mask_batch[start:end]

            # Per-viewpoint texture projection (each view gets different projection)
            tex_projected = []
            for i in range(start, end):
                vp = viewpoints[i]
                projected = self.projector.project(
                    texture,
                    yaw=vp['yaw'],
                    pitch=vp['pitch'],
                    distance=vp['distance']
                )  # [3, 1024, 1024]
                tex_projected.append(projected)

            tex_mb = torch.stack(tex_projected, dim=0)  # [bs, 3, 1024, 1024]
            tex_mb = tex_mb * mask_mb

            # Neural renderer
            rendered_mb = self.renderer.apply_differentiable(x_ref_mb, tex_mb, mask_mb)

            # Resize to detector input
            resized_mb = F.interpolate(
                rendered_mb, size=(detector_size, detector_size),
                mode='bilinear', align_corners=False
            )

            # Detector forward (pre-NMS with gradients)
            logits_mb, _ = self.detector.forward_pre_nms_with_grad(resized_mb)

            # Build per-image anchor mask from segmentation to ignore background cars.
            # Multiply car logits by a binary mask so anchors outside the padded
            # car bounding box are driven to large negative values (zero confidence).
            # The mask is detached (no grad through bbox computation), but gradients
            # still flow through the surviving anchor logits.
            anchor_boxes = self.detector.anchors.boxes.to(logits_mb.device)  # [A, 4]
            anchor_cx = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2.0     # [A]
            anchor_cy = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.0     # [A]

            car_logits = logits_mb[:, :, 2]  # [bs, A]

            # Build boolean mask: True = anchor is inside padded car bbox
            inside_masks = []
            scale_factor = detector_size / self.config['full_resolution']
            pad = 20  # pixels in detector coords
            for b in range(bs):
                seg = mask_mb[b, 0]  # [H, W] on device
                ys, xs = torch.where(seg > 0.5)
                if len(ys) == 0:
                    inside_masks.append(torch.ones(anchor_cx.shape[0],
                                                   device=logits_mb.device, dtype=torch.bool))
                    continue
                gx1 = xs.min().float() * scale_factor - pad
                gy1 = ys.min().float() * scale_factor - pad
                gx2 = xs.max().float() * scale_factor + pad
                gy2 = ys.max().float() * scale_factor + pad
                inside = (anchor_cx >= gx1) & (anchor_cx <= gx2) & \
                         (anchor_cy >= gy1) & (anchor_cy <= gy2)
                inside_masks.append(inside)

            # Apply mask: set outside anchors to large negative logit
            anchor_mask = torch.stack(inside_masks, dim=0).float()  # [bs, A]
            # Differentiable masking: keep original logits inside, force -30 outside
            # (sigmoid(-30) ≈ 0, so these contribute nothing to max_conf)
            car_logits = car_logits * anchor_mask + (-30.0) * (1.0 - anchor_mask)

            car_probs = torch.sigmoid(car_logits)
            max_conf, _ = torch.max(car_probs, dim=1)
            loss_mb = (-torch.log(1.0 - max_conf + 1e-8)).mean()

            # Scale and backward — frees this mini-batch's graph
            scaled_loss = loss_mb * (bs / num_views)
            scaled_loss.backward()

            total_loss += loss_mb.item() * bs / num_views
            all_max_conf.append(max_conf.detach().cpu())

            if return_intermediates:
                int_tex_projected.append(torch.stack(tex_projected, dim=0).detach().cpu())
                int_rendered.append(rendered_mb.detach().cpu())
                int_resized.append(resized_mb.detach().cpu())

        all_max_conf = torch.cat(all_max_conf)
        metrics = {
            'max_confidence': all_max_conf.max().item(),
            'mean_confidence': all_max_conf.mean().item(),
            'per_view_conf': all_max_conf.tolist(),
        }

        if return_intermediates:
            intermediates = {
                'tex_projected_batch': torch.cat(int_tex_projected),
                'rendered_batch': torch.cat(int_rendered),
                'rendered_resized': torch.cat(int_resized),
            }
            return total_loss, metrics, intermediates

        return total_loss, metrics

    def _tensor_to_panel(self, tensor_chw, size=512, interpolation=cv2.INTER_AREA):
        """Convert CHW RGB float [0,1] tensor to BGR uint8 resized panel."""
        img = tensor_chw.clamp(0, 1).permute(1, 2, 0).numpy()  # HWC RGB
        img_bgr = (img[:, :, ::-1] * 255).astype(np.uint8)
        return cv2.resize(img_bgr, (size, size), interpolation=interpolation)

    def _mask_to_panel(self, mask_1hw, size=512):
        """Convert 1HW float mask tensor to grayscale BGR uint8 panel."""
        mask = mask_1hw[0].clamp(0, 1).numpy()
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(
            cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR
        )
        return mask_bgr

    def _build_debug_composite(self, iteration, old_texture, new_texture,
                               viewpoints, x_ref_batch, mask_batch,
                               intermediates, metrics):
        """
        Build a single BGR composite image showing all pipeline stages.

        Layout (P*4 wide x variable height):
          Row 0: [Old Tex] [New Tex] [Diff 10x] [Info panel]
          Rows 1-N: per viewpoint [x_ref] [projected tex+mask] [mask] [det out+conf]
        """
        P = self.config.get('debug_panel_size', 512)
        num_views = x_ref_batch.shape[0]

        # --- Top row: texture evolution ---
        old_panel = self._tensor_to_panel(old_texture, P, cv2.INTER_NEAREST)
        new_panel = self._tensor_to_panel(new_texture, P, cv2.INTER_NEAREST)

        # Diff panel (amplified 10x)
        diff = (new_texture - old_texture).abs() * 10.0
        diff_panel = self._tensor_to_panel(diff.clamp(0, 1), P, cv2.INTER_NEAREST)

        # Info panel
        info_panel = np.zeros((P, P, 3), dtype=np.uint8)
        weather_name = self.current_weather['name'] if self.current_weather else 'unknown'
        spawn_idx = (self.spawn_index - 1) % self.num_spawn_points
        lines = [
            f"Iter: {iteration}",
            f"Loss: {metrics.get('loss_value', 0):.4f}",
            f"Max conf: {metrics['max_confidence']:.4f}",
            f"Mean conf: {metrics['mean_confidence']:.4f}",
            f"Spawn: {spawn_idx}  Weather: {weather_name}",
            "",
        ]
        for vi, conf in enumerate(metrics.get('per_view_conf', [])):
            vp = viewpoints[vi] if vi < len(viewpoints) else {}
            y = vp.get('yaw', 0)
            p = vp.get('pitch', 0)
            d = vp.get('distance', 0)
            lines.append(f"  V{vi} y={y:3d} p={p:2d} d={d:2d}: {conf:.4f}")

        for li, line in enumerate(lines):
            cv2.putText(info_panel, line, (10, 22 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Labels
        cv2.putText(old_panel, "Old Texture", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(new_panel, "New Texture", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(diff_panel, "Diff (10x)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        top_row = np.hstack([old_panel, new_panel, diff_panel, info_panel])

        # --- Per-viewpoint rows ---
        tex_projected_batch = intermediates['tex_projected_batch']  # [N, 3, 1024, 1024]
        rendered_resized = intermediates['rendered_resized']        # [N, 3, 512, 512]

        view_rows = []
        for vi in range(num_views):
            vp = viewpoints[vi] if vi < len(viewpoints) else {}

            # 1. Reference image with detector bounding box
            ref_panel = self._tensor_to_panel(x_ref_batch[vi].cpu(), P)
            ref_resized = F.interpolate(
                x_ref_batch[vi].unsqueeze(0), size=(512, 512),
                mode='bilinear', align_corners=False
            )
            ref_dets = self.detector.detect_cars_with_boxes(ref_resized, score_threshold=0.01)[0]
            if len(ref_dets['scores']) > 0:
                scale = P / 512.0
                x1, y1, x2, y2 = ref_dets['boxes'][0] * scale
                s = ref_dets['scores'][0]
                color = (0, 0, 255) if s > 0.5 else (0, 200, 255)
                cv2.rectangle(ref_panel, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(ref_panel, f"{s:.2f}", (int(x1), max(int(y1) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            y_lbl = vp.get('yaw', 0)
            p_lbl = vp.get('pitch', 0)
            d_lbl = vp.get('distance', 0)
            cv2.putText(ref_panel, f"Ref y={y_lbl} p={p_lbl} d={d_lbl}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # 2. Projected texture + mask overlay
            mask_hw = mask_batch[vi, 0].cpu().numpy()
            tex_panel = self._tensor_to_panel(tex_projected_batch[vi], P, cv2.INTER_NEAREST)
            mask_small = cv2.resize(mask_hw, (P, P), interpolation=cv2.INTER_NEAREST)
            mask_3ch = np.stack([mask_small] * 3, axis=-1)
            tex_panel = (tex_panel.astype(np.float32) * (0.3 + 0.7 * mask_3ch)).astype(np.uint8)
            mask_u8 = (mask_small * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tex_panel, contours, -1, (0, 0, 255), 1)
            cv2.putText(tex_panel, "Proj Tex+Mask", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # 3. Mask
            m_panel = self._mask_to_panel(mask_batch[vi].cpu(), P)
            cv2.putText(m_panel, "Mask", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 4. Detection output with bounding boxes
            det_panel = self._tensor_to_panel(rendered_resized[vi], P)
            conf = metrics.get('per_view_conf', [0] * num_views)[vi]

            det_input = rendered_resized[vi].unsqueeze(0).to(
                next(self.detector.model.parameters()).device)
            car_dets = self.detector.detect_cars_with_boxes(det_input, score_threshold=0.01)[0]

            if len(car_dets['scores']) > 0:
                scale = P / 512.0
                x1, y1, x2, y2 = car_dets['boxes'][0] * scale
                s = car_dets['scores'][0]
                color = (0, 0, 255) if s > 0.5 else (0, 200, 255)
                cv2.rectangle(det_panel, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(det_panel, f"{s:.2f}", (int(x1), max(int(y1) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            border_color = (0, 0, 255) if conf > 0.5 else (0, 255, 0)
            cv2.rectangle(det_panel, (0, 0), (P - 1, P - 1), border_color, 3)
            cv2.putText(det_panel, f"Conf: {conf:.3f}", (5, P - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2, cv2.LINE_AA)
            cv2.putText(det_panel, "Det Out", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            view_rows.append(np.hstack([ref_panel, tex_panel, m_panel, det_panel]))

        composite = np.vstack([top_row] + view_rows)
        return composite

    def _save_checkpoint(self, iteration, texture, metrics):
        """Save texture checkpoint."""
        texture_np = texture.detach().cpu().numpy()
        np.save(str(self.checkpoint_dir / f'texture_iter_{iteration:04d}.npy'), texture_np)

        # Save visualization (project at reference viewpoint)
        with torch.no_grad():
            texture_full = self.projector.project(texture, yaw=0, pitch=10, distance=8.0)
        texture_vis = texture_full.detach().cpu().numpy()
        texture_vis = np.transpose(texture_vis, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_vis, str(self.viz_dir / f'texture_iter_{iteration:04d}.png'))

        print(f"  Checkpoint saved: texture_iter_{iteration:04d}.npy")

    def _save_final_results(self, texture, training_history):
        """Save final results."""
        # Save coarse texture
        texture_np = texture.numpy()
        np.save(str(self.final_dir / 'texture_final.npy'), texture_np)

        # Save torch checkpoint
        torch.save(texture, str(self.final_dir / 'texture_final.pt'))

        # Save visualization (tiled projection at reference viewpoint)
        with torch.no_grad():
            texture_full = self.projector.project(
                texture.to(self.device), yaw=0, pitch=10, distance=8.0
            )
        texture_vis = texture_full.detach().cpu().numpy()
        texture_vis = np.transpose(texture_vis, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_vis, str(self.final_dir / 'texture_final.png'))

        # Save summary
        if len(training_history) > 0:
            last_row = training_history[-1]
            summary = {
                'training_config': self.config,
                'final_metrics': {
                    'loss': float(last_row[1]) if len(last_row) > 1 else None,
                    'max_confidence': float(last_row[2]) if len(last_row) > 2 else None,
                    'mean_confidence': float(last_row[3]) if len(last_row) > 3 else None,
                    'iterations_completed': len(training_history),
                },
                'timestamp': datetime.now().isoformat(),
                'framework': 'PyTorch (true autograd)',
                'phase': 'phase2_robust_eot',
            }
        else:
            summary = {'error': 'No training data recorded'}

        with open(self.final_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Final results saved to {self.final_dir}/")

    def train(self):
        """
        Main training loop.

        Algorithm:
            1. Initialize random 16x16 texture
            2. Every 10 iterations: change spawn point + weather, capture 24 viewpoints
            3. Per iteration: sample 12 from cached 24, forward+backward with
               per-viewpoint texture projection
            4. Adam optimizer step + clamp to [0, 1]
            5. Log metrics, checkpoint every 100 iterations
        """
        print("=" * 70)
        print("STARTING PHASE 2: ROBUST EOT TRAINING")
        print("=" * 70)
        print()

        # --- Initialize texture ---
        coarse_size = self.config['coarse_size']
        print(f"Initializing texture: random_uniform at {coarse_size}x{coarse_size}")
        texture = torch.rand(3, coarse_size, coarse_size, device=self.device)
        texture = texture.clone().requires_grad_(True)
        print(f"  Shape: {texture.shape}, requires_grad: {texture.requires_grad}")
        print(f"  Mean: {texture.mean().item():.4f}")

        # Save initial texture visualization
        with torch.no_grad():
            init_full = self.projector.project(texture, yaw=0, pitch=10, distance=8.0)
        init_vis = init_full.detach().cpu().numpy().transpose(1, 2, 0)
        visualize_texture(init_vis, str(self.viz_dir / 'texture_iter_0000.png'))
        print()

        # --- Optimizer ---
        optimizer = optim.Adam([texture], lr=self.config['learning_rate'])
        print(f"Optimizer: Adam (lr={self.config['learning_rate']})")
        print()

        # --- CSV logger ---
        logger = CSVLogger(str(self.output_dir / 'training_log.csv'))
        logger.write_header([
            'iteration', 'loss', 'max_conf', 'mean_conf',
            'grad_norm', 'texture_mean', 'texture_std',
            'spawn_idx', 'weather_name',
        ])

        # --- Initial cache fill ---
        print("Initial spawn+weather setup...")
        self._refresh_spawn_weather_cache()
        print()

        # --- Training loop ---
        print("Training Phase:")
        print("-" * 70)

        debug_active = self.config.get('debug', False)
        debug_panel_size = self.config.get('debug_panel_size', 512)
        debug_viewport_w = debug_panel_size * 4
        debug_viewport_h = 900
        if debug_active:
            print("DEBUG MODE: Showing composite visualization after each iteration")
            print("  Use scroll trackbar to pan, any key to advance, ESC to disable")
            cv2.namedWindow('EOT Debug', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('EOT Debug', debug_viewport_w, debug_viewport_h)

        views_per_iter = self.config['views_per_iteration']
        interval = self.config['spawn_weather_interval']

        for iteration in range(self.config['num_iterations']):
            # Refresh spawn + weather every N iterations
            if iteration > 0 and iteration % interval == 0:
                self._refresh_spawn_weather_cache()

            # Sample 12 viewpoints from the 24 cached
            num_cached = len(self.cached_viewpoints)
            sample_size = min(views_per_iter, num_cached)
            sample_indices = random.sample(range(num_cached), sample_size)
            iter_viewpoints = [self.cached_viewpoints[i] for i in sample_indices]
            iter_refs = self.cached_refs[sample_indices]
            iter_masks = self.cached_masks[sample_indices]

            # Save old texture for debug diff
            if debug_active:
                old_texture = texture.detach().cpu().clone()

            # Zero gradients
            optimizer.zero_grad()

            # Forward + backward (gradient accumulation inside)
            if debug_active:
                loss_value, metrics, intermediates = self._forward_pass(
                    texture, iter_viewpoints, iter_refs, iter_masks,
                    return_intermediates=True)
            else:
                loss_value, metrics = self._forward_pass(
                    texture, iter_viewpoints, iter_refs, iter_masks
                )

            # Gradient clipping
            if self.config.get('clip_grad_norm'):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [texture], self.config['clip_grad_norm']
                )
            else:
                grad_norm = texture.grad.norm().item()

            # Optimizer step
            optimizer.step()

            # Clamp texture to valid range
            with torch.no_grad():
                texture.clamp_(0.0, 1.0)

            # Debug visualization with scrollable window
            if debug_active:
                new_texture = texture.detach().cpu().clone()
                metrics['loss_value'] = loss_value
                composite = self._build_debug_composite(
                    iteration, old_texture, new_texture,
                    iter_viewpoints, iter_refs, iter_masks,
                    intermediates, metrics)
                comp_h = composite.shape[0]
                max_scroll = max(0, comp_h - debug_viewport_h)
                cv2.createTrackbar('Scroll', 'EOT Debug', 0, max(1, max_scroll), lambda x: None)
                while True:
                    y = cv2.getTrackbarPos('Scroll', 'EOT Debug')
                    y = min(y, max_scroll)
                    viewport = composite[y:y + debug_viewport_h]
                    cv2.imshow('EOT Debug', viewport)
                    key = cv2.waitKey(50) & 0xFF
                    if key == 27:  # ESC
                        print("  Debug disabled by ESC. Continuing training...")
                        cv2.destroyAllWindows()
                        debug_active = False
                        break
                    elif key != 255:  # any other key
                        break

            # Logging
            if iteration % self.config['log_every'] == 0:
                weather_name = self.current_weather['name'] if self.current_weather else 'unknown'
                spawn_idx = (self.spawn_index - 1) % self.num_spawn_points

                gn = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                print(
                    f"Iter {iteration:4d}/{self.config['num_iterations']} | "
                    f"Loss: {loss_value:.4f} | "
                    f"Max Conf: {metrics['max_confidence']:.4f} | "
                    f"Mean Conf: {metrics['mean_confidence']:.4f} | "
                    f"Grad: {gn:.6f}"
                )

                logger.write_row([
                    iteration,
                    loss_value,
                    metrics['max_confidence'],
                    metrics['mean_confidence'],
                    gn,
                    texture.mean().item(),
                    texture.std().item(),
                    spawn_idx,
                    weather_name,
                ])

            # Checkpointing
            if iteration % self.config['checkpoint_every'] == 0 and iteration > 0:
                self._save_checkpoint(iteration, texture, metrics)

        # Cleanup debug window if still open
        if self.config.get('debug', False):
            cv2.destroyAllWindows()

        # --- Final save ---
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        final_texture = texture.detach().cpu()
        self._save_final_results(final_texture, logger.data)

        logger.close()

        return {
            'texture': final_texture.numpy(),
            'final_loss': loss_value,
            'history': logger.data,
        }


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: Robust EOT Adversarial Attack')
    parser.add_argument('--debug', action='store_true',
                        help='Show composite debug visualization after each iteration '
                             '(pauses for key press)')
    parser.add_argument('--skip-load-world', action='store_true',
                        help='Connect to existing CARLA world without reloading '
                             '(use with source-built CARLA)')
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2: ROBUST EOT ADVERSARIAL TEXTURE OPTIMIZATION")
    print("=" * 70)
    print()

    CONFIG['debug'] = args.debug

    # 1. Connect to CARLA
    print("Step 1: Connecting to CARLA...")
    handler = CarlaHandler(town=CONFIG['town'], x_res=1024, y_res=1024,
                           skip_load_world=args.skip_load_world)
    handler.destroy_all_vehicles()
    handler.world_tick(30)
    time.sleep(1.0)
    handler.spawn_vehicle('vehicle.tesla.model3', color=(124, 124, 124))
    handler.world_tick(60)
    time.sleep(2.0)
    print(f"  Vehicle spawned at spawn point {handler.spawn_point_index}")
    print()

    # 2. Initialize detector
    print("Step 2: Initializing EfficientDet-D0...")
    detector = EfficientDetPyTorch()

    # 3. Initialize neural renderer
    print("Step 3: Initializing U-Net3 renderer...")
    renderer = TextureApplicatorPyTorch(
        model_path='models/unet3/trained/best_model.pt'
    )

    # 4. Initialize texture projector
    print("Step 4: Initializing texture projector...")
    projector = RepeatedTextureProjection(
        tile_count=CONFIG['tile_count'],
        reference_distance=CONFIG['reference_distance'],
        full_resolution=CONFIG['full_resolution'],
    )
    print(f"  Tile count: {CONFIG['tile_count']}, Reference distance: {CONFIG['reference_distance']}m")
    print()

    # 5. Create trainer and run
    print("Step 5: Initializing trainer...")
    trainer = RobustEOTTrainer(handler, detector, renderer, projector, CONFIG)
    print()

    print("Step 6: Starting training...")
    results = trainer.train()

    # Summary
    print()
    print("=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Output: {CONFIG['output_dir']}")
    print()

    # Cleanup
    handler.destroy_all_vehicles()
    print("Vehicles destroyed. Done.")
