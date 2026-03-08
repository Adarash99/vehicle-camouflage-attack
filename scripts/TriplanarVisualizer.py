#!/usr/bin/env python3
"""
Triplanar Visualizer - Interactive visualization with depth-based triplanar texture projection.

Uses CARLA depth sensor and TriplanarProjection for geometry-aware texture mapping.
Textures appear painted onto the car surface with correct 3D tiling.

Run via: edit run.sh launch line to `cd $ROOT/scripts && python TriplanarVisualizer.py`
"""

import sys
import os
import time

import numpy as np
import cv2
import torch

# Add project root for models.* and attack.* imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CarlaHandler import CarlaHandler
from texture_applicator_pytorch import TextureApplicatorPyTorch
from attack.texture_projection import TriplanarProjection


class TriplanarVisualizer:

    WINDOW_NAME = "Triplanar Visualizer"
    FULL_RES = 1024

    def __init__(self, town='Town10HD'):
        # Initialize CARLA
        self.handler = CarlaHandler(town=town)
        time.sleep(5)
        self.handler.spawn_vehicle('vehicle.tesla.model3', color=(124, 124, 124))

        # Initialize renderer
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'models', 'unet3', 'trained', 'best_model.pt')
        self.applicator = TextureApplicatorPyTorch(model_path=model_path)

        # Triplanar projection (default tile_size=0.5m)
        self.projector = TriplanarProjection(
            tile_size=0.5, resolution=self.FULL_RES, fov=90.0, blend_sharpness=4.0
        )

        # Texture state
        self.texture_mode = 'solid'
        self.texture_small = None  # 8x8 base pattern (numpy HWC float32)
        self._generate_texture()

        # Control state
        self.yaw = 0
        self.pitch = 15
        self.distance = 10
        self.spawn_index = 0
        self.view_mode = 1  # 0=top, 1=3d
        self.tile_size_slider = 50  # Maps to 0.5m

        # Debug mode: color-code projection planes
        self.debug_mode = False
        self.debug_overlay = None  # [H, W, 3] uint8 RGB plane map

        # Display cache
        self.rendered_display = np.zeros((self.FULL_RES, self.FULL_RES, 3), dtype=np.uint8)

        # Dirty flag for re-rendering
        self._dirty = True

        # Warm up sensors (including depth)
        self.handler.world_tick(30)

        # Create window and trackbars
        self._setup_window()

        # Initial render
        self._render()

    def _generate_texture(self):
        """Generate a solid random color texture."""
        self.texture_color = np.random.randint(0, 256, size=3).astype(np.float32) / 255.0
        self.texture_mode = 'solid'
        self._dirty = True

    def _generate_random_texture(self):
        """Generate a coarse 8x8 random block texture (projected via triplanar at render time)."""
        self.texture_small = np.random.rand(8, 8, 3).astype(np.float32)
        self.texture_mode = 'random'
        self._dirty = True

    def _setup_window(self):
        """Create OpenCV window with trackbars."""
        cv2.namedWindow(self.WINDOW_NAME)

        num_spawns = self.handler.get_spawn_points()

        # Camera controls
        cv2.createTrackbar("View", self.WINDOW_NAME, 1, 1, self._on_view)
        cv2.createTrackbar("Distance", self.WINDOW_NAME, 10, 100, self._on_distance)
        cv2.createTrackbar("Yaw", self.WINDOW_NAME, 0, 360, self._on_yaw)
        cv2.createTrackbar("Pitch", self.WINDOW_NAME, 15, 90, self._on_pitch)
        cv2.createTrackbar("Spawn", self.WINDOW_NAME, 0, max(num_spawns - 1, 0), self._on_spawn)

        # Triplanar-specific control
        cv2.createTrackbar("Tile Size", self.WINDOW_NAME, 50, 200, self._on_tile_size)

        # Mouse callback for button clicks
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    # --- Trackbar callbacks ---

    def _on_view(self, val):
        self.view_mode = val
        self._dirty = True

    def _on_distance(self, val):
        self.distance = max(val, 1)
        self._dirty = True

    def _on_yaw(self, val):
        self.yaw = val
        self._dirty = True

    def _on_pitch(self, val):
        self.pitch = val
        self._dirty = True

    def _on_spawn(self, val):
        if val != self.spawn_index:
            self.spawn_index = val
            self.handler.change_spawn_point(val)
            self._dirty = True

    def _on_tile_size(self, val):
        self.tile_size_slider = max(val, 20)  # Minimum 20 = 0.2m
        # Map slider 20-200 to 0.2m-2.0m
        self.projector.tile_size = self.tile_size_slider / 100.0
        if self.texture_mode == 'random':
            self._dirty = True

    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks on texture mode buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            btn_x1 = self.FULL_RES - 280
            btn_x2 = self.FULL_RES - 10
            # Solid button
            if btn_x1 <= x <= btn_x2 and self.FULL_RES - 110 <= y <= self.FULL_RES - 65:
                self._generate_texture()
                self._render()
            # Random button
            elif btn_x1 <= x <= btn_x2 and self.FULL_RES - 60 <= y <= self.FULL_RES - 10:
                self._generate_random_texture()
                self._render()

    def _render(self):
        """Full render pipeline: CARLA capture -> triplanar projection -> U-Net3."""
        # Update CARLA camera
        self.handler.spectator_view = 'top' if self.view_mode == 0 else '3d'
        self.handler.spectator_distance = self.distance
        self.handler.spectator_pitch = self.pitch
        self.handler.spectator_yaw = self.yaw
        self.handler._update_spectator()

        # Tick for fresh sensor data
        self.handler.world_tick(20)

        # Get reference image (BGR uint8 from CARLA)
        ref_bgr = self.handler.get_image()  # [1024, 1024, 3] uint8 BGR
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        ref_float = ref_rgb.astype(np.float32) / 255.0

        # Get car mask
        mask = self.handler.get_car_segmentation_mask()  # [1024, 1024] float32

        # Create texture masked to car pixels
        if self.texture_mode == 'solid':
            masked_texture = np.zeros((self.FULL_RES, self.FULL_RES, 3), dtype=np.float32)
            masked_texture[mask > 0.5] = self.texture_color
        else:  # random — project 8x8 texture via triplanar projection
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tex_t = torch.from_numpy(self.texture_small.transpose(2, 0, 1)).float().to(device)

            # Get depth and transforms from CARLA
            depth = self.handler.get_depth()
            cam_transform = self.handler.get_camera_transform()
            veh_transform = self.handler.get_vehicle_transform()

            with torch.no_grad():
                projected = self.projector.project(
                    tex_t, depth, mask, cam_transform, veh_transform
                )  # [3, 1024, 1024]

            projected_np = projected.cpu().numpy().transpose(1, 2, 0)  # -> HWC
            masked_texture = projected_np  # Already masked inside project()

            # Compute debug plane map if debug mode is active
            if self.debug_mode:
                self.debug_overlay = self.projector.get_debug_plane_map(
                    depth, mask, cam_transform, veh_transform
                )
            else:
                self.debug_overlay = None

        # Run U-Net3 renderer (expects RGB, outputs RGB)
        rendered_float = self.applicator.apply(ref_float, masked_texture, mask)

        # Convert RGB output back to BGR for cv2.imshow
        rendered_uint8 = np.clip(rendered_float * 255, 0, 255).astype(np.uint8)
        self.rendered_display = cv2.cvtColor(rendered_uint8, cv2.COLOR_RGB2BGR)
        self._dirty = False

    def _draw_button(self, image):
        """Draw texture mode buttons and swatch on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        btn_x1 = self.FULL_RES - 280
        btn_x2 = self.FULL_RES - 10

        # Solid button (upper)
        solid_y1, solid_y2 = self.FULL_RES - 110, self.FULL_RES - 65
        solid_border = (0, 255, 0) if self.texture_mode == 'solid' else (200, 200, 200)
        cv2.rectangle(image, (btn_x1, solid_y1), (btn_x2, solid_y2), (80, 80, 80), -1)
        cv2.rectangle(image, (btn_x1, solid_y1), (btn_x2, solid_y2), solid_border, 2)
        cv2.putText(image, "Solid [N]", (btn_x1 + 40, solid_y2 - 10),
                    font, 0.9, (255, 255, 255), 2)

        # Random button (lower)
        rand_y1, rand_y2 = self.FULL_RES - 60, self.FULL_RES - 10
        rand_border = (0, 255, 0) if self.texture_mode == 'random' else (200, 200, 200)
        cv2.rectangle(image, (btn_x1, rand_y1), (btn_x2, rand_y2), (80, 80, 80), -1)
        cv2.rectangle(image, (btn_x1, rand_y1), (btn_x2, rand_y2), rand_border, 2)
        cv2.putText(image, "Random [R]", (btn_x1 + 30, rand_y2 - 15),
                    font, 0.9, (255, 255, 255), 2)

        # Swatch
        sw_x1, sw_y1 = 10, self.FULL_RES - 60
        sw_x2, sw_y2 = 60, self.FULL_RES - 10
        if self.texture_mode == 'solid':
            r, g, b = (self.texture_color * 255).astype(int)
            cv2.rectangle(image, (sw_x1, sw_y1), (sw_x2, sw_y2),
                          (int(b), int(g), int(r)), -1)
        else:
            preview = cv2.resize(self.texture_small, (50, 50), interpolation=cv2.INTER_NEAREST)
            preview_bgr = (preview[:, :, ::-1] * 255).astype(np.uint8)
            image[sw_y1:sw_y2, sw_x1:sw_x2] = preview_bgr
        cv2.rectangle(image, (sw_x1, sw_y1), (sw_x2, sw_y2), (200, 200, 200), 2)

        # Tile size label
        tile_m = self.projector.tile_size
        cv2.putText(image, f"Tile: {tile_m:.2f}m", (10, self.FULL_RES - 70),
                    font, 0.6, (255, 255, 255), 1)

    def run(self):
        """Main display loop."""
        print("Triplanar Visualizer running. ESC=exit, N=solid, R=random, D=debug planes")

        while True:
            if self._dirty:
                self._render()

            display = self.rendered_display.copy()

            # Show debug plane overlay if active
            if self.debug_mode and self.debug_overlay is not None:
                # Convert RGB debug overlay to BGR for OpenCV
                debug_bgr = cv2.cvtColor(self.debug_overlay, cv2.COLOR_RGB2BGR)
                # Blend overlay with rendered image (50/50 where overlay is non-zero)
                overlay_mask = (self.debug_overlay.sum(axis=-1) > 0)
                display[overlay_mask] = (
                    display[overlay_mask].astype(np.float32) * 0.4
                    + debug_bgr[overlay_mask].astype(np.float32) * 0.6
                ).astype(np.uint8)
                # DEBUG label
                cv2.putText(display, "DEBUG", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            self._draw_button(display)

            cv2.imshow(self.WINDOW_NAME, display)
            key = cv2.waitKey(50) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('n'):
                self._generate_texture()
                self._render()
            elif key == ord('r'):
                self._generate_random_texture()
                self._render()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                self._dirty = True

        cv2.destroyAllWindows()


if __name__ == '__main__':
    town = 'Town10HD'
    if len(sys.argv) > 1:
        town = sys.argv[1]

    try:
        viz = TriplanarVisualizer(town=town)
        viz.run()
    finally:
        cv2.destroyAllWindows()
