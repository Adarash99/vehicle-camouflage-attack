#!/usr/bin/env python3
"""
Probe Script: Test CARLA Texture Streaming API on Spawned Vehicles

Tests whether CARLA 0.9.14's runtime texture streaming API can apply
textures to spawned vehicle meshes (not just static world objects).

What it does:
1. Connect to CARLA (with rendering enabled)
2. Spawn Tesla Model 3
3. Discover all object names (filter for vehicle/tesla keywords)
4. Try applying a bright solid-color texture to candidate names
5. Capture camera images before/after to verify visual change
6. Report which object names work (if any)

Usage:
    # Edit run.sh to launch this script, then:
    ./run.sh
"""

import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
from scripts.CarlaHandler import CarlaHandler


def make_solid_texture(width, height, r, g, b):
    """Create a solid-color carla.TextureColor."""
    texture = carla.TextureColor(width, height)
    for x in range(width):
        for y in range(height):
            texture.set(x, y, carla.Color(r, g, b, 255))
    return texture


def numpy_to_carla_texture(image_np):
    """Convert numpy HxWx3 uint8 image to carla.TextureColor (Y-flipped)."""
    h, w = image_np.shape[:2]
    texture = carla.TextureColor(w, h)
    for x in range(w):
        for y in range(h):
            r, g, b = int(image_np[y, x, 0]), int(image_np[y, x, 1]), int(image_np[y, x, 2])
            texture.set(x, h - y - 1, carla.Color(r, g, b, 255))
    return texture


def image_diff(img1, img2):
    """Compute mean absolute pixel difference between two images."""
    return np.abs(img1.astype(float) - img2.astype(float)).mean()


def main():
    print("=" * 70)
    print("PROBE: CARLA Texture Streaming API on Vehicles")
    print("=" * 70)
    print()

    # --- Step 1: Connect to CARLA with rendering enabled ---
    print("Step 1: Connecting to CARLA (rendering enabled)...")
    handler = CarlaHandler(town='Town05', x_res=1024, y_res=1024)

    # CRITICAL: Disable no_rendering_mode so textures actually render
    settings = handler.world.get_settings()
    settings.no_rendering_mode = False
    handler.world.apply_settings(settings)
    print("  no_rendering_mode = False (rendering enabled)")

    # --- Step 2: Spawn Tesla Model 3 ---
    print("\nStep 2: Spawning Tesla Model 3...")
    handler.spawn_vehicle('vehicle.tesla.model3', color=(124, 124, 124))
    handler.world_tick(50)
    time.sleep(1.0)

    # Position camera for clear view
    handler.set_camera_viewpoint(yaw=30, pitch=-15, distance=8)
    handler.world_tick(30)
    time.sleep(0.5)

    # Capture baseline image
    baseline_img = handler.get_image()
    os.makedirs('experiments/probe_texture_api', exist_ok=True)
    cv2.imwrite('experiments/probe_texture_api/01_baseline.png', baseline_img)
    print(f"  Baseline image saved (shape: {baseline_img.shape})")

    # --- Step 3: Discover object names ---
    print("\nStep 3: Discovering object names...")
    all_names = handler.world.get_names_of_all_objects()
    print(f"  Total objects in world: {len(all_names)}")

    # Filter for vehicle-related keywords
    keywords = ['tesla', 'vehicle', 'car', 'SM_', 'model3', 'sedan', 'Body', 'body',
                'Paint', 'paint', 'Chassis', 'chassis', 'mesh', 'Mesh']
    filtered = {}
    for kw in keywords:
        matches = [n for n in all_names if kw.lower() in n.lower()]
        if matches:
            filtered[kw] = matches

    print(f"\n  Filtered results:")
    for kw, names in filtered.items():
        print(f"    '{kw}': {len(names)} matches")
        for name in names[:10]:  # show first 10
            print(f"      - {name}")
        if len(names) > 10:
            print(f"      ... and {len(names) - 10} more")

    # Save all names for reference
    with open('experiments/probe_texture_api/all_object_names.txt', 'w') as f:
        for name in sorted(all_names):
            f.write(name + '\n')
    print(f"\n  All {len(all_names)} object names saved to all_object_names.txt")

    # --- Step 4: Try applying texture to candidate names ---
    print("\nStep 4: Testing texture application...")

    # Build candidate list: all unique names from filtered results
    candidates = set()
    for names in filtered.values():
        candidates.update(names)
    candidates = sorted(candidates)

    # Also try the vehicle actor's type_id as a name (unlikely but worth trying)
    if handler.vehicle is not None:
        actor_name = handler.vehicle.type_id
        candidates.insert(0, actor_name)
        # Also try the actor ID
        candidates.insert(0, f"Vehicle_{handler.vehicle.id}")
        print(f"  Vehicle actor: type_id={actor_name}, id={handler.vehicle.id}")

    print(f"  Testing {len(candidates)} candidate names...")

    # Create a bright green texture (should be very obvious if applied)
    tex_size = 512
    green_tex = make_solid_texture(tex_size, tex_size, 0, 255, 0)
    empty_float_tex = carla.TextureFloatColor(0, 0)

    successful_names = []
    tested_count = 0

    for name in candidates:
        tested_count += 1
        if tested_count > 50:  # Don't test too many
            print(f"  Stopping after {tested_count - 1} candidates (remaining: {len(candidates) - tested_count + 1})")
            break

        try:
            # Try apply_color_texture_to_object (simpler API)
            handler.world.apply_color_texture_to_object(
                name,
                carla.MaterialParameter.Diffuse,
                green_tex
            )
            handler.world_tick(10)
            time.sleep(0.3)

            # Capture and compare
            after_img = handler.get_image()
            diff = image_diff(baseline_img, after_img)

            if diff > 5.0:  # Significant visual change
                successful_names.append((name, diff, 'apply_color_texture_to_object'))
                cv2.imwrite(f'experiments/probe_texture_api/success_{name.replace("/", "_")}.png', after_img)
                print(f"    SUCCESS: '{name}' (diff={diff:.1f})")
            else:
                # Also try apply_textures_to_object (batch API)
                handler.world.apply_textures_to_object(
                    name, green_tex, empty_float_tex, empty_float_tex, empty_float_tex
                )
                handler.world_tick(10)
                time.sleep(0.3)

                after_img2 = handler.get_image()
                diff2 = image_diff(baseline_img, after_img2)

                if diff2 > 5.0:
                    successful_names.append((name, diff2, 'apply_textures_to_object'))
                    cv2.imwrite(f'experiments/probe_texture_api/success_{name.replace("/", "_")}.png', after_img2)
                    print(f"    SUCCESS: '{name}' (diff={diff2:.1f}) via apply_textures_to_object")

        except Exception as e:
            # Some names may cause errors — that's expected
            pass

    # --- Step 5: Report results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if successful_names:
        print(f"\n  {len(successful_names)} object name(s) worked!")
        for name, diff, method in successful_names:
            print(f"    Name: '{name}'")
            print(f"    Method: {method}")
            print(f"    Visual diff: {diff:.1f}")
            print()

        # Apply a pattern texture to the best candidate for a final demo
        best_name = successful_names[0][0]
        print(f"  Applying checkerboard pattern to '{best_name}' for final verification...")

        # Create a colorful checkerboard pattern
        checker = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
        cell = tex_size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    checker[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = [255, 0, 0]  # Red
                else:
                    checker[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = [0, 0, 255]  # Blue

        pattern_tex = numpy_to_carla_texture(checker)
        handler.world.apply_color_texture_to_object(
            best_name, carla.MaterialParameter.Diffuse, pattern_tex
        )
        handler.world_tick(30)
        time.sleep(0.5)

        pattern_img = handler.get_image()
        cv2.imwrite('experiments/probe_texture_api/05_pattern_applied.png', pattern_img)
        print("  Pattern applied and captured.")

        # Orbit camera to capture multiple angles
        print("\n  Capturing multiple viewpoints with pattern applied...")
        for yaw in range(0, 360, 45):
            handler.set_camera_viewpoint(yaw=yaw, pitch=-15, distance=8)
            handler.world_tick(20)
            time.sleep(0.3)
            img = handler.get_image()
            cv2.imwrite(f'experiments/probe_texture_api/pattern_yaw_{yaw:03d}.png', img)
            print(f"    yaw={yaw:3d} captured")

    else:
        print("\n  NO object names worked for texture application on vehicles.")
        print("  The texture streaming API may not support spawned vehicle actors.")
        print("\n  Fallback: Use the neural renderer for visualization (as during training).")
        print("  The TextureEvaluator will use neural-rendered output for evaluation.")

    # Save results summary
    with open('experiments/probe_texture_api/results.txt', 'w') as f:
        f.write("Probe Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total objects: {len(all_names)}\n")
        f.write(f"Candidates tested: {tested_count}\n")
        f.write(f"Successful: {len(successful_names)}\n\n")
        for name, diff, method in successful_names:
            f.write(f"Name: {name}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Diff: {diff:.1f}\n\n")

    # Cleanup
    handler.destroy_all_vehicles()
    print("\nDone. Check experiments/probe_texture_api/ for images and results.")


if __name__ == '__main__':
    main()
