#!/usr/bin/env python3
"""Test if CARLA semantic segmentation works on the static Tesla mesh.

Connects to CARLA, finds the static Tesla, orbits the camera around it
at every degree from 0 to 359, and saves both RGB and segmentation mask
images side by side. Includes mask cleanup to remove small disconnected
patches (e.g. distant background cars).
"""

import sys
import os
import time
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
from scripts.CarlaHandler import CarlaHandler


def clean_segmentation_mask(seg_mask):
    """Keep only the largest connected component in the segmentation mask.

    Args:
        seg_mask: [H, W] float32 mask (1=car, 0=background)

    Returns:
        Cleaned mask with only the largest connected component.
    """
    mask_uint8 = (seg_mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        # Only background (label 0), no car pixels
        return seg_mask

    # Find the largest component (skip label 0 = background)
    # stats[:, cv2.CC_STAT_AREA] gives area for each label
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_label = np.argmax(areas) + 1  # +1 because we skipped background

    cleaned = np.zeros_like(seg_mask)
    cleaned[labels == largest_label] = 1.0
    return cleaned


def orbit_camera(handler, location, yaw, pitch, distance):
    """Position spectator camera orbiting a world location."""
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    x_offset = math.cos(yaw_rad) * math.cos(pitch_rad) * distance
    y_offset = math.sin(yaw_rad) * math.cos(pitch_rad) * distance
    z_offset = -math.sin(pitch_rad) * distance

    camera_pos = carla.Location(
        x=location.x - x_offset,
        y=location.y - y_offset,
        z=location.z + z_offset + 1.5,
    )
    camera_rot = carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
    handler.spectator.set_transform(carla.Transform(camera_pos, camera_rot))
    handler.world.tick()


def main():
    print("Connecting to CARLA...")
    handler = CarlaHandler(x_res=1024, y_res=1024, no_rendering=False, skip_load_world=True)

    # Find static Tesla
    print("Finding static Tesla mesh...")
    name_lower = "sm_teslam3_v2"
    static_location = None
    static_name = None
    try:
        for label in [carla.CityObjectLabel.Car, carla.CityObjectLabel.Any]:
            env_objects = handler.world.get_environment_objects(label)
            for obj in env_objects:
                if name_lower in obj.name.lower():
                    static_location = obj.transform.location
                    static_name = obj.name
                    print(f"  Found: {static_name} at ({static_location.x:.1f}, {static_location.y:.1f}, {static_location.z:.1f})")
                    break
            if static_location:
                break
    except Exception as e:
        print(f"  Error: {e}")

    if static_location is None:
        print("  Could not find static Tesla mesh!")
        return

    # Capture all 360 viewpoints
    output_dir = "test_scripts/static_seg_output"
    os.makedirs(output_dir, exist_ok=True)

    missing_yaws = []
    cleaned_yaws = []  # viewpoints where small patches were removed

    for yaw in range(0, 360):
        orbit_camera(handler, static_location, yaw, -15, 8)
        handler.world_tick(20)
        time.sleep(0.15)

        # Get RGB image
        rgb_img = handler.get_image()

        # Get segmentation mask
        handler.world_tick(5)
        seg_mask_raw = handler.get_car_segmentation_mask()  # [H, W] float32, 1=car

        # Clean mask: keep only the largest connected component
        seg_mask = clean_segmentation_mask(seg_mask_raw)

        raw_pixels = int(seg_mask_raw.sum())
        car_pixels = int(seg_mask.sum())
        removed = raw_pixels - car_pixels
        total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
        pct = 100 * car_pixels / total_pixels

        if removed > 0:
            cleaned_yaws.append((yaw, removed))

        if car_pixels > 0:
            ys, xs = np.where(seg_mask > 0.5)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            missing_yaws.append(yaw)

        # Create side-by-side: RGB | Segmentation mask (colorized)
        mask_vis = np.zeros_like(rgb_img)
        mask_vis[seg_mask > 0.5] = (0, 255, 0)  # Green for car pixels (cleaned)
        # Show removed pixels in red
        removed_mask = (seg_mask_raw > 0.5) & (seg_mask < 0.5)
        mask_vis[removed_mask] = (0, 0, 255)  # Red for removed patches

        composite = np.hstack([rgb_img, mask_vis])

        # Add labels
        label_text = f"RGB (yaw={yaw})"
        cv2.putText(composite, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        mask_text = f"Seg mask ({car_pixels} px, {pct:.1f}%)"
        if removed > 0:
            mask_text += f" [removed {removed} px]"
        cv2.putText(composite, mask_text, (1034, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(f"{output_dir}/yaw_{yaw:03d}.png", composite)

        status = "OK" if car_pixels > 0 else "NO MASK"
        extra = f" (removed {removed} stray px)" if removed > 0 else ""
        print(f"  Yaw {yaw:3d}: {car_pixels:6d} px ({pct:4.1f}%) [{status}]{extra}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Total viewpoints: 360")
    print(f"  With segmentation: {360 - len(missing_yaws)}")
    print(f"  Missing segmentation: {len(missing_yaws)}")
    if missing_yaws:
        ranges = []
        start = missing_yaws[0]
        end = missing_yaws[0]
        for y in missing_yaws[1:]:
            if y == end + 1:
                end = y
            else:
                ranges.append((start, end))
                start = y
                end = y
        ranges.append((start, end))
        print(f"  Missing ranges: {', '.join(f'{s}-{e}' if s != e else str(s) for s, e in ranges)}")

    print(f"\n  Viewpoints with stray patches removed: {len(cleaned_yaws)}")
    if cleaned_yaws:
        for yaw, removed in cleaned_yaws:
            print(f"    Yaw {yaw:3d}: removed {removed} pixels")

    print(f"\nImages saved to {output_dir}/")
    print(f"  Green = kept (largest component), Red = removed (stray patches)")


if __name__ == '__main__':
    main()
