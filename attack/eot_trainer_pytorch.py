#!/usr/bin/env python3
"""
Pure PyTorch EOT (Expectation Over Transformation) Trainer

Trains adversarial camouflage textures across multiple viewpoints using
TRUE end-to-end gradient flow through a pure PyTorch pipeline.

Key Improvements over TensorFlow version:
- No finite differences - uses native PyTorch autograd
- Single forward + backward pass per iteration (3x faster)
- True gradient flow: Texture -> Renderer -> Detector -> Loss
- Simpler code without numpy bridges

Architecture:
    Texture (PyTorch) -> Renderer (PyTorch) -> Detector (PyTorch) -> Loss (PyTorch)
    Gradients computed via true backpropagation through entire pipeline.

Usage:
    from attack.eot_trainer_pytorch import EOTTrainerPyTorch

    trainer = EOTTrainerPyTorch(
        carla_handler=carla,
        detector=detector,
        renderer=renderer,
        viewpoints=viewpoints,
        config=config
    )
    results = trainer.train()

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

import os
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

from attack.loss_pytorch import attack_loss_pytorch, attack_loss_with_stats_pytorch
from attack.logger import CSVLogger


def create_viewpoint_configs():
    """
    Returns standard 12-viewpoint EOT configuration.

    Viewpoints arranged in 30 degree increments around vehicle (0-330).
    Pitch (-15) and distance (8m) are set in the trainer config.
    """
    return [{'yaw': y} for y in range(0, 360, 30)]


def visualize_texture(texture_np, save_path):
    """Save texture as PNG for visual inspection."""
    texture_uint8 = (texture_np * 255).astype(np.uint8)
    texture_bgr = cv2.cvtColor(texture_uint8, cv2.COLOR_RGB2BGR)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), texture_bgr)


# Default training configuration
DEFAULT_CONFIG = {
    'learning_rate': 0.01,
    'num_iterations': 1000,
    'checkpoint_every': 100,
    'log_every': 10,
    'optimizer': 'adam',
    'output_dir': 'experiments/phase1_eot_pytorch/',
    'coarse_size': 8,  # Texture parameterization size (8x8 -> full_res)
    'full_resolution': 1024,  # 1024x1024 for V2 renderer
    'detector_input_size': 512,  # EfficientDet expects 512x512
    'clip_grad_norm': 1.0,  # Gradient clipping for stability
}


class EOTTrainerPyTorch:
    """
    Pure PyTorch EOT trainer for adversarial textures.

    Uses TRUE end-to-end gradient flow instead of finite differences:
    - Forward: texture -> renderer -> detector -> loss
    - Backward: loss.backward() computes all gradients via autograd
    - Update: optimizer.step() updates texture

    This is 3x faster than the finite differences approach and provides
    more accurate gradients.
    """

    def __init__(self, carla_handler, detector, renderer, viewpoints, config=None):
        """
        Initialize PyTorch EOT trainer.

        Args:
            carla_handler: CarlaHandler instance (must have vehicle spawned)
            detector: EfficientDetPyTorch instance
            renderer: TextureApplicatorPyTorch instance
            viewpoints: List of viewpoint dicts from create_viewpoint_configs()
            config: Training configuration dict (merged with DEFAULT_CONFIG)
        """
        self.carla = carla_handler
        self.detector = detector
        self.renderer = renderer
        self.viewpoints = viewpoints
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Detect device
        self.device = next(detector.model.parameters()).device
        print(f"EOTTrainerPyTorch using device: {self.device}")

        # Validate inputs
        if self.carla.vehicle is None:
            raise ValueError("CarlaHandler must have a vehicle spawned")

        self.spawn_point_interval = self.config.get('spawn_point_interval', 10)

        # Set vehicle to neutral grey and configure camera once
        #self.carla.change_vehicle_color((124, 124, 124))
        #self.carla.world.tick(30)
        self.carla.update_view('3d')
        self.carla.update_pitch(self.config.get('pitch', 15))
        self.carla.update_distance(self.config.get('distance', 8))
        self.carla.update_yaw(-1)  # dummy yaw so first real update triggers move
        self.carla.world.tick(30)

        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.viz_dir = self.output_dir / 'visualizations'
        self.final_dir = self.output_dir / 'final'

        self.checkpoint_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        self.final_dir.mkdir(exist_ok=True)

        # Save configuration
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"EOTTrainerPyTorch initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Viewpoints: {len(self.viewpoints)}")
        print(f"  Device: {self.device}")

    def capture_reference_images(self):
        """
        Capture reference images and masks from all viewpoints.

        Returns:
            x_ref_batch: torch.Tensor [6, 3, H, W] float32 [0, 1] (NCHW)
            mask_batch: torch.Tensor [6, 1, H, W] float32 [0, 1] (NCHW)
        """
        resolution = self.config['full_resolution']
        print(f"Capturing reference images from {len(self.viewpoints)} viewpoints...")

        x_refs = []
        masks = []
        
        

        for i, vp in enumerate(self.viewpoints):
            # Position camera (only yaw changes; pitch/distance set in __init__)
            self.carla.update_yaw(vp['yaw'])
            self.carla.world.tick(30)

            time.sleep(0.5)  # seconds to wait after moving camera

            # Capture image (BGR uint8)
            img = self.carla.get_image()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0  # [H, W, 3]

            # Capture car segmentation mask (matches U-Net3 training data)
            self.carla.world.tick(20)  # ensure seg_labels is fresh
            car_mask = self.carla.get_car_segmentation_mask()  # [H, W] float32

            x_refs.append(img_float)
            masks.append(car_mask)

            print(f"  Viewpoint {i}: yaw={vp['yaw']:3d}")

            # Save for debugging
            save_path = self.output_dir / f'reference_view_{i}_yaw_{vp["yaw"]:03d}.png'
            visualize_texture(img_float, str(save_path))

        # Stack and convert to tensors (NHWC -> NCHW)
        x_ref_np = np.stack(x_refs, axis=0)  # [6, H, W, 3]
        x_ref_np = np.transpose(x_ref_np, (0, 3, 1, 2))  # [6, 3, H, W]
        x_ref_batch = torch.from_numpy(x_ref_np).float().to(self.device)

        mask_np = np.stack(masks, axis=0)[:, :, :, np.newaxis]  # [6, H, W, 1]
        mask_np = np.transpose(mask_np, (0, 3, 1, 2))  # [6, 1, H, W]
        mask_batch = torch.from_numpy(mask_np).float().to(self.device)

        print(f"  Reference batch: {x_ref_batch.shape}")
        print(f"  Mask batch: {mask_batch.shape}")

        return x_ref_batch, mask_batch

    def initialize_texture(self, init_type='random_uniform'):
        """
        Initialize texture as torch.Tensor with requires_grad=True.

        Args:
            init_type: Initialization strategy
                - 'random_uniform': Random values in [0, 1]
                - 'random_normal': Normal distribution clipped to [0, 1]
                - 'constant': Constant gray (0.5)

        Returns:
            torch.Tensor [3, H, W] with requires_grad=True (NCHW format)
        """
        coarse_size = self.config['coarse_size']
        print(f"Initializing texture: {init_type} at {coarse_size}x{coarse_size}")

        if init_type == 'random_uniform':
            texture_init = torch.rand(3, coarse_size, coarse_size, device=self.device)
        elif init_type == 'random_normal':
            texture_init = torch.randn(3, coarse_size, coarse_size, device=self.device) * 0.1 + 0.5
            texture_init = torch.clamp(texture_init, 0.0, 1.0)
        elif init_type == 'constant':
            texture_init = torch.ones(3, coarse_size, coarse_size, device=self.device) * 0.5
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        # Enable gradients for optimization
        texture = texture_init.clone().requires_grad_(True)

        print(f"  Shape: {texture.shape}")
        print(f"  requires_grad: {texture.requires_grad}")
        print(f"  Mean: {texture.mean().item():.4f}")

        # Save initial texture
        texture_full = self._upsample_texture(texture)
        texture_np = texture_full.detach().cpu().numpy()
        texture_np = np.transpose(texture_np, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_np, str(self.viz_dir / 'texture_iter_0000.png'))

        return texture

    def _upsample_texture(self, texture):
        """
        Upsample coarse texture to full resolution.

        Args:
            texture: [3, coarse_size, coarse_size] or [B, 3, coarse_size, coarse_size]

        Returns:
            [3, full_res, full_res] or [B, 3, full_res, full_res]
        """
        full_res = self.config['full_resolution']
        coarse_size = self.config['coarse_size']
        tile_count = self.config.get('tile_count', coarse_size)  # default 8

        squeeze = False
        if texture.dim() == 3:
            texture = texture.unsqueeze(0)
            squeeze = True

        # Tile then upsample: [B,3,8,8] → [B,3,64,64] → [B,3,1024,1024]
        tiled = texture.repeat(1, 1, tile_count, tile_count)
        upsampled = F.interpolate(tiled, size=(full_res, full_res), mode='nearest')

        if squeeze:
            upsampled = upsampled.squeeze(0)

        return upsampled

    def _tile_texture_for_batch(self, texture):
        """
        Repeat single texture for all viewpoints.

        Args:
            texture: [3, H, W]

        Returns:
            [batch_size, 3, H, W]
        """
        batch_size = len(self.viewpoints)
        return texture.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _forward_pass(self, texture, x_ref_batch, mask_batch, return_intermediates=False):
        """
        Forward pass through renderer + detector, processing viewpoints in
        mini-batches with gradient accumulation to fit in GPU memory.

        Calls .backward() per mini-batch so computation graphs are freed
        immediately. Gradients accumulate on `texture`.

        Args:
            texture: torch.Tensor [3, coarse_size, coarse_size] (requires_grad=True)
            x_ref_batch: torch.Tensor [N, 3, H, W] reference images
            mask_batch: torch.Tensor [N, 1, H, W] paintable masks
            return_intermediates: If True, return detached CPU copies of intermediates

        Returns:
            loss_value: float (detached, for logging only — gradients already accumulated)
            metrics: Dict with confidence values (detached)
            intermediates: (only if return_intermediates=True) Dict of detached CPU tensors
        """
        full_res = self.config['full_resolution']
        tile_count = self.config.get('tile_count', self.config['coarse_size'])
        detector_size = self.config['detector_input_size']
        mini_batch = self.config.get('views_per_batch', 4)
        num_views = x_ref_batch.shape[0]

        all_max_conf = []
        total_loss = 0.0
        int_rendered = [] if return_intermediates else None
        int_resized = [] if return_intermediates else None

        for start in range(0, num_views, mini_batch):
            end = min(start + mini_batch, num_views)
            bs = end - start

            x_ref_mb = x_ref_batch[start:end]
            mask_mb = mask_batch[start:end]

            # Tile and upsample texture (recomputed per mini-batch for fresh graph)
            tiled = texture.repeat(1, tile_count, tile_count)
            texture_full = F.interpolate(
                tiled.unsqueeze(0), size=(full_res, full_res), mode='nearest'
            ).squeeze(0)  # [3, 1024, 1024]

            tex_mb = texture_full.unsqueeze(0).expand(bs, -1, -1, -1)
            tex_mb = tex_mb * mask_mb

            rendered_mb = self.renderer.apply_differentiable(x_ref_mb, tex_mb, mask_mb)

            resized_mb = F.interpolate(
                rendered_mb, size=(detector_size, detector_size),
                mode='bilinear', align_corners=False
            )

            logits_mb, _ = self.detector.forward_pre_nms_with_grad(resized_mb)

            car_logits = logits_mb[:, :, 2]
            car_probs = torch.sigmoid(car_logits)
            max_conf, _ = torch.max(car_probs, dim=1)
            loss_mb = (-torch.log(1.0 - max_conf + 1e-8)).mean()

            # Scale and backward — frees this mini-batch's graph
            scaled_loss = loss_mb * (bs / num_views)
            scaled_loss.backward()

            total_loss += loss_mb.item() * bs / num_views
            all_max_conf.append(max_conf.detach().cpu())

            if return_intermediates:
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
                'texture_full': texture_full.detach().cpu(),
                'rendered_batch': torch.cat(int_rendered),
                'rendered_resized': torch.cat(int_resized),
            }
            return total_loss, metrics, intermediates

        return total_loss, metrics

    def _tensor_to_panel(self, tensor_chw, size=256, interpolation=cv2.INTER_AREA):
        """Convert CHW RGB float [0,1] tensor to BGR uint8 resized panel."""
        img = tensor_chw.clamp(0, 1).permute(1, 2, 0).numpy()  # HWC RGB
        img_bgr = (img[:, :, ::-1] * 255).astype(np.uint8)
        return cv2.resize(img_bgr, (size, size), interpolation=interpolation)

    def _mask_to_panel(self, mask_1hw, size=256):
        """Convert 1HW float mask tensor to grayscale BGR uint8 panel."""
        mask = mask_1hw[0].clamp(0, 1).numpy()
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        return mask_bgr

    def _build_debug_composite(self, iteration, old_texture, new_texture,
                               x_ref_batch, mask_batch, intermediates, metrics):
        """
        Build a single BGR composite image showing all pipeline stages.

        Layout (1280 x variable height):
          Row 0: [Old Tex 256x256] [New Tex 256x256] [Diff 10x 256x256] [Info 256x512]
          Rows 1-N: per viewpoint [x_ref] [tex+mask overlay] [mask] [U-Net out] [det out+conf]
        """
        P = self.config.get('debug_panel_size', 512)  # panel size
        num_views = x_ref_batch.shape[0]

        # --- Top row: texture evolution ---
        old_panel = self._tensor_to_panel(old_texture, P, cv2.INTER_NEAREST)
        new_panel = self._tensor_to_panel(new_texture, P, cv2.INTER_NEAREST)

        # Diff panel (amplified 10x)
        diff = (new_texture - old_texture).abs() * 10.0
        diff_panel = self._tensor_to_panel(diff.clamp(0, 1), P, cv2.INTER_NEAREST)

        # Info panel (P high x P wide)
        info_panel = np.zeros((P, P, 3), dtype=np.uint8)
        lines = [
            f"Iter: {iteration}",
            f"Loss: {metrics.get('loss_value', 0):.4f}",
            f"Max conf: {metrics['max_confidence']:.4f}",
            f"Mean conf: {metrics['mean_confidence']:.4f}",
            "",
        ]
        for vi, conf in enumerate(metrics['per_view_conf']):
            yaw = self.viewpoints[vi]['yaw'] if vi < len(self.viewpoints) else 0
            lines.append(f"  View {vi} (yaw={yaw:3d}): {conf:.4f}")

        for li, line in enumerate(lines):
            cv2.putText(info_panel, line, (10, 22 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Labels for top row
        cv2.putText(old_panel, "Old Texture", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(new_panel, "New Texture", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(diff_panel, "Diff (10x)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        top_row = np.hstack([old_panel, new_panel, diff_panel, info_panel])  # 256 x 1280

        # --- Per-viewpoint rows ---
        texture_full = intermediates['texture_full']   # [3, 1024, 1024]
        rendered_batch = intermediates['rendered_batch']  # [N, 3, 1024, 1024]
        rendered_resized = intermediates['rendered_resized']  # [N, 3, 512, 512]

        view_rows = []
        for vi in range(num_views):
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
            cv2.putText(ref_panel, f"Ref (yaw={self.viewpoints[vi]['yaw']})", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 2. Texture + mask overlay
            mask_hw = mask_batch[vi, 0].cpu().numpy()  # [H, W]
            tex_panel = self._tensor_to_panel(texture_full, P, cv2.INTER_NEAREST)
            # Darken areas outside mask
            mask_small = cv2.resize(mask_hw, (P, P), interpolation=cv2.INTER_NEAREST)
            mask_3ch = np.stack([mask_small] * 3, axis=-1)
            tex_panel = (tex_panel.astype(np.float32) * (0.3 + 0.7 * mask_3ch)).astype(np.uint8)
            # Draw mask contour in red
            mask_u8 = (mask_small * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tex_panel, contours, -1, (0, 0, 255), 1)
            # Draw 8x8 grid lines in green
            coarse = self.config['coarse_size']
            step = P // coarse
            for g in range(1, coarse):
                cv2.line(tex_panel, (g * step, 0), (g * step, P), (0, 128, 0), 1)
                cv2.line(tex_panel, (0, g * step), (P, g * step), (0, 128, 0), 1)
            cv2.putText(tex_panel, "Tex+Mask", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 3. Mask
            m_panel = self._mask_to_panel(mask_batch[vi].cpu(), P)
            cv2.putText(m_panel, "Mask", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 4. Detection output (rendered, resized) with bounding boxes
            det_panel = self._tensor_to_panel(rendered_resized[vi], P)
            conf = metrics['per_view_conf'][vi]

            # Run detector to get decoded bounding boxes for this view
            det_input = rendered_resized[vi].unsqueeze(0).to(next(self.detector.model.parameters()).device)
            car_dets = self.detector.detect_cars_with_boxes(det_input, score_threshold=0.01)[0]

            # Draw only the single highest-confidence car bounding box
            if len(car_dets['scores']) > 0:
                scale = P / 512.0
                x1, y1, x2, y2 = car_dets['boxes'][0] * scale
                s = car_dets['scores'][0]
                color = (0, 0, 255) if s > 0.5 else (0, 200, 255)
                cv2.rectangle(det_panel, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(det_panel, f"{s:.2f}", (int(x1), max(int(y1) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            # Red border if detected (conf > 0.5), green if fooled
            border_color = (0, 0, 255) if conf > 0.5 else (0, 255, 0)
            cv2.rectangle(det_panel, (0, 0), (P - 1, P - 1), border_color, 3)
            # Confidence text
            label = f"Conf: {conf:.3f}"
            cv2.putText(det_panel, label, (5, P - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2, cv2.LINE_AA)
            cv2.putText(det_panel, "Det Out", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            view_rows.append(np.hstack([ref_panel, tex_panel, m_panel, det_panel]))

        composite = np.vstack([top_row] + view_rows)
        return composite

    def train(self):
        """
        Main EOT training loop using true PyTorch autograd.

        Algorithm:
            1. Capture reference images from all viewpoints
            2. Initialize texture as trainable parameter
            3. Setup PyTorch optimizer
            4. For each iteration:
                a. Forward pass (texture -> renderer -> detector -> loss)
                b. Backward pass (loss.backward() computes all gradients)
                c. Optimizer step (update texture)
                d. Clamp texture to [0, 1]
                e. Log metrics
            5. Save final results

        Returns:
            dict: {'texture': final texture, 'final_loss': loss, 'history': log data}
        """
        print("=" * 70)
        print("STARTING PYTORCH EOT TRAINING")
        print("=" * 70)
        print()

        # === SETUP ===
        print("Setup Phase:")
        print("-" * 70)

        # 1. Capture reference images
        x_ref_batch, mask_batch = self.capture_reference_images()
        print()

        # 2. Initialize texture
        texture = self.initialize_texture('random_uniform')
        print()

        # 3. Setup optimizer
        optimizer = optim.Adam([texture], lr=self.config['learning_rate'])
        print(f"Optimizer: Adam (lr={self.config['learning_rate']})")
        print()

        # 4. Initialize loggers
        num_views = len(self.viewpoints)
        view_headers = [f'view_{i}_conf' for i in range(num_views)]
        main_logger = CSVLogger(str(self.output_dir / 'training_log.csv'))
        main_logger.write_header([
            'iteration', 'loss', 'max_conf', 'mean_conf',
            'grad_norm', 'texture_mean', 'texture_std',
        ] + view_headers)

        # === TRAINING LOOP ===
        print("Training Phase:")
        print("-" * 70)

        debug_active = self.config.get('debug', False)
        debug_viewport_h = 900  # visible height in pixels (fits most screens)
        if debug_active:
            print("DEBUG MODE: Showing composite visualization after each iteration")
            print("  Use scroll trackbar to pan, any key to advance, ESC to disable")
            cv2.namedWindow('EOT Debug', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('EOT Debug', 1280, debug_viewport_h)

        for iteration in range(self.config['num_iterations']):
            # Change spawn point every N iterations for scene diversity
            if iteration > 0 and iteration % self.spawn_point_interval == 0:
                num_spawns = len(self.carla.spawn_points)
                new_idx = random.randint(0, num_spawns - 1)
                print(f"  Changing spawn point to {new_idx}...")
                self.carla.change_spawn_point(new_idx)
                self.carla.world.tick(30)
                x_ref_batch, mask_batch = self.capture_reference_images()

            # Save old texture for debug diff
            if debug_active:
                old_texture = texture.detach().cpu().clone()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with gradient accumulation (backward called per mini-batch)
            if debug_active:
                loss_value, metrics, intermediates = self._forward_pass(
                    texture, x_ref_batch, mask_batch, return_intermediates=True)
            else:
                loss_value, metrics = self._forward_pass(texture, x_ref_batch, mask_batch)

            # Gradient clipping for stability
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
                    x_ref_batch, mask_batch, intermediates, metrics)
                comp_h = composite.shape[0]
                max_scroll = max(0, comp_h - debug_viewport_h)
                cv2.createTrackbar('Scroll', 'EOT Debug', 0, max(1, max_scroll), lambda x: None)
                # Scroll loop: trackbar pans, any key advances, ESC quits debug
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
                print(
                    f"Iter {iteration:4d}/{self.config['num_iterations']} | "
                    f"Loss: {loss_value:.4f} | "
                    f"Max Conf: {metrics['max_confidence']:.4f} | "
                    f"Mean Conf: {metrics['mean_confidence']:.4f} | "
                    f"Grad: {grad_norm:.6f}"
                )

                main_logger.write_row([
                    iteration,
                    loss_value,
                    metrics['max_confidence'],
                    metrics['mean_confidence'],
                    grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    texture.mean().item(),
                    texture.std().item(),
                ] + metrics['per_view_conf'])

            # Checkpointing
            if iteration % self.config['checkpoint_every'] == 0 and iteration > 0:
                self._save_checkpoint(iteration, texture, metrics)

        # Cleanup debug window if still open
        if self.config.get('debug', False):
            cv2.destroyAllWindows()

        # === FINAL SAVE ===
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        final_texture = texture.detach().cpu()
        self._save_final_results(final_texture, main_logger.data)

        main_logger.close()

        return {
            'texture': final_texture.numpy(),
            'final_loss': loss_value,
            'history': main_logger.data,
        }

    def _save_checkpoint(self, iteration, texture, metrics):
        """Save texture checkpoint."""
        # Save numpy (for resuming)
        texture_np = texture.detach().cpu().numpy()
        np.save(str(self.checkpoint_dir / f'texture_iter_{iteration:04d}.npy'), texture_np)

        # Save visualization
        texture_full = self._upsample_texture(texture)
        texture_vis = texture_full.detach().cpu().numpy()
        texture_vis = np.transpose(texture_vis, (1, 2, 0))  # CHW -> HWC
        visualize_texture(texture_vis, str(self.viz_dir / f'texture_iter_{iteration:04d}.png'))

        print(f"  Checkpoint saved: texture_iter_{iteration:04d}.npy")

    def _save_final_results(self, texture, training_history):
        """Save final results."""
        full_res = self.config['full_resolution']

        # Save coarse texture
        texture_np = texture.numpy()
        np.save(str(self.final_dir / 'texture_final.npy'), texture_np)

        # Save torch checkpoint
        torch.save(texture, str(self.final_dir / 'texture_final.pt'))

        # Save visualization (tiled then upsampled)
        texture_full = self._upsample_texture(texture)
        texture_vis = texture_full.numpy()
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
            }
        else:
            summary = {'error': 'No training data recorded'}

        with open(self.final_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Final results saved to {self.final_dir}/")


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH EOT TRAINER TEST")
    print("=" * 70)
    print()

    # Test viewpoint configs
    print("Testing create_viewpoint_configs...")
    vps = create_viewpoint_configs()
    assert len(vps) == 6
    print(f"  Created {len(vps)} viewpoints")
    print()

    # Test texture operations
    print("Testing texture operations...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create test texture
    texture = torch.rand(3, 128, 128, device=device, requires_grad=True)
    print(f"  Coarse texture: {texture.shape}")

    # Test upsampling (using function directly)
    texture_up = F.interpolate(
        texture.unsqueeze(0),
        size=(1024, 1024),
        mode='nearest',
    ).squeeze(0)
    print(f"  Upsampled texture: {texture_up.shape}")

    # Test gradient flow
    loss = texture_up.mean()
    loss.backward()
    print(f"  Gradient norm: {texture.grad.norm().item():.6f}")

    if texture.grad.norm().item() > 0:
        print("  PASSED: Gradients flow through upsampling!")
    print()

    # Note: Full trainer test requires CARLA, detector, and renderer
    print("Note: Full trainer test requires:")
    print("  - CARLA server running")
    print("  - EfficientDetPyTorch instance")
    print("  - TextureApplicatorPyTorch instance")
    print()
    print("Run experiments/phase1_random_pytorch.py for full test")

    print()
    print("=" * 70)
    print("TRAINER TESTS PASSED")
    print("=" * 70)
