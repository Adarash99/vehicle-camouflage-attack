#!/usr/bin/env python3
"""
Renderer Visualizer - Interactive visualization of U-Net3 rendering pipeline.

Displays rendered output at full 1024x1024 resolution with interactive camera sliders.

Run via: edit run.sh launch line to `cd $ROOT/scripts && python RendererVisualizer.py`
"""

import sys
import os
import time

import numpy as np
import cv2
import torch

# Add project root for models.* and attack.* imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F

from CarlaHandler import CarlaHandler
from texture_applicator_pytorch import TextureApplicatorPyTorch


class RendererVisualizer:

    WINDOW_NAME = "Renderer Visualizer"
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

        # Texture state
        self.texture_mode = 'solid'
        self.texture_small = None  # 8x8 base pattern (numpy HWC float32)
        self._generate_texture()

        # Tiling controls (manual, not auto-updated from camera)
        self.proj_x_shift = 0.0
        self.proj_y_shift = 0.0
        self.proj_scale = 1.0

        # Control state
        self.yaw = 0
        self.pitch = 15
        self.distance = 10
        self.spawn_index = 0
        self.view_mode = 1  # 0=top, 1=3d

        # Display cache
        self.rendered_display = np.zeros((self.FULL_RES, self.FULL_RES, 3), dtype=np.uint8)

        # Dirty flag for re-rendering
        self._dirty = True

        # Warm up sensors
        self.handler.world_tick(30)

        # Create window and trackbars
        self._setup_window()

        # Initial render
        self._render()

    def _generate_texture(self):
        """Generate a solid random color texture (matches dataset_8k_revised distribution)."""
        self.texture_color = np.random.randint(0, 256, size=3).astype(np.float32) / 255.0
        self.texture_mode = 'solid'
        self._dirty = True

    def _generate_random_texture(self):
        """Generate a coarse 8x8 random block texture (projected at render time)."""
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

        # Texture projection controls
        cv2.createTrackbar("X Shift", self.WINDOW_NAME, 300, 600, self._on_x_shift)
        cv2.createTrackbar("Y Shift", self.WINDOW_NAME, 100, 200, self._on_y_shift)
        cv2.createTrackbar("Scale", self.WINDOW_NAME, 100, 400, self._on_proj_scale)

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

    def _on_x_shift(self, val):
        self.proj_x_shift = (val - 300) / 50.0
        if self.texture_mode == 'random':
            self._dirty = True

    def _on_y_shift(self, val):
        self.proj_y_shift = (val - 100) / 50.0
        if self.texture_mode == 'random':
            self._dirty = True

    def _on_proj_scale(self, val):
        self.proj_scale = max(val, 10) / 100.0
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
        """Full render pipeline: CARLA capture -> solid color texture -> U-Net3."""
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
        else:  # random — tile 8x8 texture with scale/shift controls
            tex_t = torch.from_numpy(self.texture_small.transpose(2, 0, 1)).float().unsqueeze(0)  # [1, 3, 8, 8]
            res = self.FULL_RES
            scale = self.proj_scale

            # Build sampling grid with scale and shift
            coords = torch.linspace(-1, 1, res)
            grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
            grid = grid * scale
            grid[..., 0] += self.proj_x_shift
            grid[..., 1] += self.proj_y_shift
            # Wrap around for seamless tiling
            grid = torch.remainder(grid + 1, 2) - 1

            tiled = F.grid_sample(tex_t, grid, mode='nearest', padding_mode='border', align_corners=True)
            tiled_np = tiled.squeeze(0).numpy().transpose(1, 2, 0)  # [H, W, 3]

            masked_texture = tiled_np * mask[:, :, np.newaxis]

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

    def run(self):
        """Main display loop."""
        print("Renderer Visualizer running. Press ESC to exit, N for solid texture, R for random texture.")

        while True:
            if self._dirty:
                self._render()

            display = self.rendered_display.copy()
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

        cv2.destroyAllWindows()


if __name__ == '__main__':
    town = 'Town10HD'
    if len(sys.argv) > 1:
        town = sys.argv[1]

    try:
        viz = RendererVisualizer(town=town)
        viz.run()
    finally:
        cv2.destroyAllWindows()
