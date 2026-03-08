#!/usr/bin/env python3
"""
Comparison Evaluator: Side-by-side neural renderer vs CARLA direct rendering.

Runs two evaluation pipelines in parallel for each viewpoint:
1. Neural renderer — capture gray car reference + mask → U-Net3 render with texture → EfficientDet
2. CARLA direct — orbit camera around static mesh (texture applied in UE4 editor) → EfficientDet

This reveals how well the neural renderer approximates real CARLA rendering,
producing a side-by-side video and CSV with per-viewpoint detection confidences.

Usage:
    python evaluation-scripts/compare_evaluator.py \
        --object-name SM_TeslaM3_parked \
        --texture experiments/phase1_random/final/texture_final.npy

Output:
    evaluation/evalN/
    ├── detection_results.csv   (yaw, neural_conf, carla_conf)
    ├── evaluation.mp4          (side-by-side video)
    └── yaw_NNN.png             (individual frames)
"""

import sys
import os
import time
import csv
import math
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # evaluation-scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import carla
import torch
import torch.nn.functional as F

from scripts.CarlaHandler import CarlaHandler
from scripts.texture_applicator_pytorch import TextureApplicatorPyTorch
from attack.detector_pytorch import EfficientDetPyTorch

# Reuse utilities from TextureEvaluator
from TextureEvaluator import (
    orbit_camera,
    find_object_location,
    load_texture,
    tile_and_upsample,
    draw_detections,
    filter_target_detections,
    filter_by_gt_bbox,
    clean_segmentation_mask,
    mask_to_bbox,
    add_confidence_graphs,
    save_video,
    save_csv,
)


def find_next_eval_dir(base_dir):
    """Find the next available evalN/ directory under base_dir.

    Returns:
        Path to the next eval directory (e.g. evaluation/eval3/)
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"eval{n}").exists():
        n += 1
    eval_dir = base / f"eval{n}"
    eval_dir.mkdir()
    return eval_dir


def _mask_iou(m1, m2):
    """Compute IoU between two binary masks."""
    b1 = m1 > 0.5
    b2 = m2 > 0.5
    inter = (b1 & b2).sum()
    union = (b1 | b2).sum()
    return inter / union if union > 0 else 0.0


def calibrate_yaw_offset(handler, static_location, viewpoints):
    """Find the yaw offset between the spawned vehicle and the static mesh.

    Captures a mask from the static mesh at yaw=0, then sweeps the spawned
    vehicle camera offset to find the best mask IoU match.

    Args:
        handler: CarlaHandler with a spawned vehicle
        static_location: carla.Location of the static mesh
        viewpoints: list of viewpoint dicts (used for pitch/distance defaults)

    Returns:
        (best_offset, best_iou): The calibrated yaw offset in degrees and its IoU
    """
    vehicle_loc = handler.vehicle.get_transform().location
    test_yaw = 0
    test_pitch = viewpoints[0].get('pitch', -15)
    test_dist = viewpoints[0].get('distance', 8)

    # Capture reference mask from the static mesh
    orbit_camera(handler, static_location, test_yaw, test_pitch, test_dist)
    handler.world_tick(20)
    time.sleep(0.3)
    static_mask = clean_segmentation_mask(handler.get_car_segmentation_mask())

    # Coarse search: 5° steps across full 360°
    best_offset = 0.0
    best_iou = 0.0
    for offset in range(-180, 181, 5):
        neural_yaw = (test_yaw + offset) % 360
        orbit_camera(handler, vehicle_loc, neural_yaw, test_pitch, test_dist)
        handler.world_tick(10)
        time.sleep(0.15)
        neural_mask = clean_segmentation_mask(handler.get_car_segmentation_mask())
        iou = _mask_iou(static_mask, neural_mask)
        if iou > best_iou:
            best_iou = iou
            best_offset = offset

    # Fine search: 1° steps around best coarse offset
    coarse_best = best_offset
    for offset in range(int(coarse_best) - 5, int(coarse_best) + 6):
        neural_yaw = (test_yaw + offset) % 360
        orbit_camera(handler, vehicle_loc, neural_yaw, test_pitch, test_dist)
        handler.world_tick(10)
        time.sleep(0.15)
        neural_mask = clean_segmentation_mask(handler.get_car_segmentation_mask())
        iou = _mask_iou(static_mask, neural_mask)
        if iou > best_iou:
            best_iou = iou
            best_offset = offset

    return best_offset, best_iou


def evaluate_comparison(handler, detector, renderer, texture_chw,
                        viewpoints, static_location, output_dir, device,
                        neural_yaw_offset, spawn_indices=None):
    """Run side-by-side evaluation for each viewpoint.

    For each yaw angle:
      - Neural path: position camera on spawned vehicle → capture ref + mask →
        neural render → detect
      - CARLA path: orbit camera around static mesh → capture frame → detect

    If a broken mask is detected (< 1% of image pixels), the vehicle is
    respawned at a different random spawn point and recalibrated.

    Args:
        handler: CarlaHandler with a spawned gray vehicle
        detector: EfficientDetPyTorch instance
        renderer: TextureApplicatorPyTorch instance
        texture_chw: [3, H, W] float32 texture in [0, 1]
        viewpoints: list of dicts with yaw/pitch/distance
        static_location: carla.Location of the static mesh
        output_dir: Path for output
        device: torch device
        neural_yaw_offset: Calibrated yaw offset (degrees) for the spawned vehicle
        spawn_indices: list of remaining spawn indices to try if respawn needed

    Returns:
        results: list of dicts with per-viewpoint data
        frames: list of BGR composite frames
    """
    resolution = 1024
    det_size = 512
    coarse_size = texture_chw.shape[1]
    MIN_MASK_RATIO = 0.01  # Mask must cover at least 1% of image pixels
    MIN_ALIGN_IOU = 0.5

    # Tile and upsample texture once
    texture_full = tile_and_upsample(texture_chw, tile_count=coarse_size, resolution=resolution)
    texture_full_t = torch.from_numpy(texture_full).float().to(device)  # [3, 1024, 1024]

    vehicle_loc = handler.vehicle.get_transform().location
    remaining_spawns = list(spawn_indices) if spawn_indices else []

    results = []
    frames = []

    for i, vp in enumerate(viewpoints):
        yaw = vp['yaw']
        pitch = vp.get('pitch', -15)
        distance = vp.get('distance', 8)

        # ---- Neural renderer path ----
        # Apply yaw offset so neural view matches CARLA static mesh view
        neural_yaw = (yaw + neural_yaw_offset) % 360
        orbit_camera(handler, vehicle_loc, neural_yaw, pitch, distance)
        handler.world_tick(30)
        time.sleep(0.5)

        # Capture reference image and mask
        img_bgr = handler.get_image()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        handler.world_tick(20)
        car_mask = handler.get_car_segmentation_mask()  # [H, W]

        # Check for broken mask — respawn if needed
        mask_ratio = car_mask.sum() / (resolution * resolution)
        if mask_ratio < MIN_MASK_RATIO:
            print(f"  View {i:2d} | yaw={yaw:3d} | BROKEN MASK ({mask_ratio:.4f}), respawning...")
            respawned = False
            handler.destroy_all_vehicles()
            time.sleep(0.5)
            for sp_idx in remaining_spawns[:]:
                try:
                    handler.spawn_point_index = sp_idx
                    handler.spawn_vehicle('vehicle.tesla.model3', color=(124, 124, 124))
                except RuntimeError:
                    remaining_spawns.remove(sp_idx)
                    continue
                handler.world_tick(50)
                time.sleep(1.0)
                best_offset, best_iou = calibrate_yaw_offset(
                    handler, static_location, viewpoints)
                if best_iou >= MIN_ALIGN_IOU:
                    print(f"    Respawned at spawn point {sp_idx} (IoU={best_iou:.4f})")
                    neural_yaw_offset = best_offset
                    vehicle_loc = handler.vehicle.get_transform().location
                    remaining_spawns.remove(sp_idx)
                    respawned = True
                    break
                else:
                    handler.destroy_all_vehicles()
                    remaining_spawns.remove(sp_idx)
                    time.sleep(0.5)
            if not respawned:
                print(f"    ERROR: No more spawn points to try, skipping view {i}")
                continue
            # Retry this viewpoint with new spawn
            neural_yaw = (yaw + neural_yaw_offset) % 360
            orbit_camera(handler, vehicle_loc, neural_yaw, pitch, distance)
            handler.world_tick(30)
            time.sleep(0.5)
            img_bgr = handler.get_image()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            handler.world_tick(20)
            car_mask = handler.get_car_segmentation_mask()

        # Convert to tensors [1, C, H, W]
        x_ref_t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_t = torch.from_numpy(car_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        tex_t = texture_full_t.unsqueeze(0) * mask_t

        # Neural render
        with torch.no_grad():
            rendered_t = renderer.apply_differentiable(x_ref_t, tex_t, mask_t)

        # Detect on neural rendered image
        rendered_resized = F.interpolate(rendered_t, size=(det_size, det_size),
                                         mode='bilinear', align_corners=False)
        neural_det = detector.detect_cars_with_boxes(rendered_resized, score_threshold=0.01)[0]
        neural_det = filter_target_detections(neural_det, det_size)
        neural_conf = neural_det['scores'][0] if len(neural_det['scores']) > 0 else 0.0

        # Convert neural rendered image to BGR for display
        rendered_np = rendered_t[0].detach().cpu().permute(1, 2, 0).numpy()
        neural_bgr = np.ascontiguousarray((rendered_np[:, :, ::-1] * 255).astype(np.uint8))

        # ---- CARLA direct path ----
        # Orbit camera around the static mesh
        orbit_camera(handler, static_location, yaw, pitch, distance)
        handler.world_tick(20)
        time.sleep(0.3)

        # Capture actual CARLA render
        carla_bgr = handler.get_image()
        carla_rgb = cv2.cvtColor(carla_bgr, cv2.COLOR_BGR2RGB)
        carla_float = carla_rgb.astype(np.float32) / 255.0

        # Get segmentation mask for GT bounding box filtering
        handler.world_tick(10)
        carla_seg_mask = handler.get_car_segmentation_mask()
        carla_seg_mask = clean_segmentation_mask(carla_seg_mask)
        carla_gt_box = mask_to_bbox(carla_seg_mask, image_size=resolution, det_size=det_size)

        # Detect on CARLA frame
        carla_t = torch.from_numpy(carla_float).permute(2, 0, 1).unsqueeze(0).float().to(device)
        carla_resized = F.interpolate(carla_t, size=(det_size, det_size),
                                      mode='bilinear', align_corners=False)
        carla_det = detector.detect_cars_with_boxes(carla_resized, score_threshold=0.01)[0]
        carla_det = filter_by_gt_bbox(carla_det, carla_gt_box, padding=20)
        carla_conf = carla_det['scores'][0] if len(carla_det['scores']) > 0 else 0.0

        # Record results
        results.append({
            'viewpoint': i,
            'yaw': yaw,
            'pitch': pitch,
            'distance': distance,
            'neural_conf': float(neural_conf),
            'carla_conf': float(carla_conf),
            'neural_num_det': len(neural_det['scores']),
            'carla_num_det': len(carla_det['scores']),
        })

        # ---- Build side-by-side frame ----
        # Draw detections
        draw_detections(neural_bgr, neural_det, det_size)
        draw_detections(carla_bgr, carla_det, det_size)

        # Add labels
        cv2.putText(neural_bgr, f"Neural Renderer (yaw={yaw})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(neural_bgr, f"Conf: {neural_conf:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(carla_bgr, f"CARLA Direct (yaw={yaw})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(carla_bgr, f"Conf: {carla_conf:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Side-by-side: [Neural | CARLA]
        composite = np.hstack([neural_bgr, carla_bgr])
        frames.append(composite)

        # Save individual frame
        cv2.imwrite(str(output_dir / f'yaw_{yaw:03d}.png'), composite)

        print(f"  View {i:2d} | yaw={yaw:3d} | neural={neural_conf:.4f} | carla={carla_conf:.4f}")

    return results, frames


def main():
    parser = argparse.ArgumentParser(description='Comparison Evaluator: Neural vs CARLA')
    parser.add_argument('--texture', type=str,
                        default='experiments/phase1_random/final/texture_final.npy',
                        help='Path to texture checkpoint (.npy or .pt)')
    parser.add_argument('--object-name', type=str, default='SM_TeslaM3_v2',
                        help='CARLA object name for the static vehicle mesh '
                             '(placed in UE4 editor)')
    parser.add_argument('--yaw-step', type=int, default=5,
                        help='Yaw angle step in degrees (default: 5 = 72 viewpoints)')
    parser.add_argument('--pitch', type=float, default=-15,
                        help='Camera pitch angle')
    parser.add_argument('--distance', type=float, default=8,
                        help='Camera distance from vehicle')
    parser.add_argument('--output-dir', type=str, default='evaluation/',
                        help='Base output directory (auto-increments evalN/ subdirs)')
    parser.add_argument('--renderer-path', type=str,
                        default='models/unet3/trained/best_model.pt',
                        help='Path to neural renderer weights')
    parser.add_argument('--town', type=str, default='Town01',
                        help='CARLA town (used for spawn points, not map loading)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Video frame rate')
    parser.add_argument('--cloudiness', type=float, default=None,
                        help='Override cloudiness (0-100)')
    args = parser.parse_args()

    print("=" * 70)
    print("COMPARISON EVALUATOR: Neural Renderer vs CARLA Direct")
    print("=" * 70)
    print()

    # Build viewpoint list
    viewpoints = [{'yaw': y, 'pitch': args.pitch, 'distance': args.distance}
                  for y in range(0, 360, args.yaw_step)]
    print(f"Viewpoints: {len(viewpoints)} (every {args.yaw_step} degrees)")

    # Load texture
    print(f"\nLoading texture: {args.texture}")
    texture_chw = load_texture(args.texture)

    # Connect to CARLA (skip_load_world=True for editor mode, rendering enabled)
    print(f"\nConnecting to CARLA (skip_load_world=True, rendering enabled)...")
    handler = CarlaHandler(town=args.town, x_res=1024, y_res=1024,
                           no_rendering=False, skip_load_world=True)

    # Override weather if requested
    if args.cloudiness is not None:
        handler.update_cloudiness(args.cloudiness)
        print(f"  Cloudiness set to {args.cloudiness}")

    # Find static mesh location
    print(f"\nLocating static mesh '{args.object_name}'...")
    object_name_full, static_location = find_object_location(handler, args.object_name)
    if static_location is None:
        print(f"  ERROR: Could not find '{args.object_name}' in the world.")
        print(f"  Try running: python evaluation-scripts/TextureEvaluator.py "
              f"--discover-objects {args.object_name}")
        return

    print(f"  Found: {object_name_full}")
    print(f"  Location: ({static_location.x:.1f}, {static_location.y:.1f}, {static_location.z:.1f})")

    # Spawn a gray vehicle and calibrate yaw alignment with the static mesh.
    # Try multiple spawn points until we find one with good mask overlap.
    print("\nSpawning gray vehicle for neural renderer...")
    MIN_ALIGN_IOU = 0.5
    aligned = False

    import random
    num_spawn_points = handler.get_spawn_points()
    spawn_indices = list(range(num_spawn_points))
    random.shuffle(spawn_indices)

    for sp_idx in spawn_indices:
        try:
            handler.spawn_point_index = sp_idx
            handler.spawn_vehicle('vehicle.tesla.model3', color=(124, 124, 124))
        except RuntimeError:
            continue
        handler.world_tick(50)
        time.sleep(1.0)
        print(f"  Trying spawn point {sp_idx} (of {num_spawn_points})...")

        # Calibrate yaw offset via mask IoU
        best_offset, best_iou = calibrate_yaw_offset(
            handler, static_location, viewpoints)

        if best_iou >= MIN_ALIGN_IOU:
            print(f"  Spawn point {sp_idx}: offset={best_offset:.0f}°, IoU={best_iou:.4f} — OK")
            spawn_indices.remove(sp_idx)
            aligned = True
            break
        else:
            print(f"  Spawn point {sp_idx}: offset={best_offset:.0f}°, IoU={best_iou:.4f} — too low, trying next")
            spawn_indices.remove(sp_idx)
            handler.destroy_all_vehicles()
            time.sleep(0.5)

    if not aligned:
        print("  ERROR: Could not find a spawn point with good mask alignment.")
        return

    # Initialize detector and renderer
    print("\nInitializing EfficientDet-D0...")
    detector = EfficientDetPyTorch()
    device = next(detector.model.parameters()).device

    print(f"Initializing neural renderer: {args.renderer_path}")
    renderer = TextureApplicatorPyTorch(model_path=args.renderer_path)

    # Create output directory
    output_dir = find_next_eval_dir(args.output_dir)
    print(f"\nOutput directory: {output_dir}")

    # Run comparison evaluation
    print(f"\n{'=' * 70}")
    print("RUNNING COMPARISON")
    print(f"{'=' * 70}")

    results, frames = evaluate_comparison(
        handler, detector, renderer, texture_chw,
        viewpoints, static_location, output_dir, device,
        neural_yaw_offset=best_offset,
        spawn_indices=spawn_indices
    )

    # Save results
    save_csv(results, output_dir / 'detection_results.csv')
    neural_confs = [r['neural_conf'] for r in results]
    carla_confs = [r['carla_conf'] for r in results]
    frames = add_confidence_graphs(frames, carla_confs)
    save_video(frames, output_dir / 'evaluation.mp4', fps=args.fps)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Neural Renderer:")
    print(f"    Max conf:  {max(neural_confs):.4f}")
    print(f"    Mean conf: {np.mean(neural_confs):.4f}")
    print(f"    Views > 0.5: {sum(1 for c in neural_confs if c > 0.5)}/{len(neural_confs)}")
    print(f"    Views < 0.1: {sum(1 for c in neural_confs if c < 0.1)}/{len(neural_confs)}")
    print()
    print(f"  CARLA Direct:")
    print(f"    Max conf:  {max(carla_confs):.4f}")
    print(f"    Mean conf: {np.mean(carla_confs):.4f}")
    print(f"    Views > 0.5: {sum(1 for c in carla_confs if c > 0.5)}/{len(carla_confs)}")
    print(f"    Views < 0.1: {sum(1 for c in carla_confs if c < 0.1)}/{len(carla_confs)}")
    print()

    # Correlation between neural and CARLA
    correlation = np.corrcoef(neural_confs, carla_confs)[0, 1]
    mean_diff = np.mean(np.abs(np.array(neural_confs) - np.array(carla_confs)))
    print(f"  Correlation (neural vs CARLA): {correlation:.4f}")
    print(f"  Mean absolute difference:      {mean_diff:.4f}")

    # Cleanup
    handler.destroy_all_vehicles()

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
