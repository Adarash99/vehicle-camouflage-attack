#!/usr/bin/env python3
"""
Transfer Evaluation: Test adversarial texture transferability across detectors.

Runs a chosen detector on a pre-placed static vehicle mesh in CARLA with the
adversarial texture applied via the texture streaming API. This tests whether
textures optimized against EfficientDet-D0 transfer to other detectors.

Supported detectors:
  - efficientdet: EfficientDet-D0 (same as training target)
  - yolov5s: YOLOv5-small (transfer target)
  - yolov5m: YOLOv5-medium
  - yolov5l: YOLOv5-large
  - ssd: SSD300-VGG16
  - faster_rcnn: Faster R-CNN (ResNet50-FPN)
  - mask_rcnn: Mask R-CNN (ResNet50-FPN)

Usage:
    # YOLOv5 transfer test on static mesh
    python evaluation/transfer_eval.py --detector yolov5s \
        --object-name SM_TeslaM3_v2 \
        --texture experiments/phase1_random/final/texture_final.npy

    # Faster R-CNN transfer test
    python evaluation/transfer_eval.py --detector faster_rcnn \
        --object-name SM_TeslaM3_v2 \
        --texture experiments/phase1_random/final/texture_final.npy

    # EfficientDet baseline (same detector as training)
    python evaluation/transfer_eval.py --detector efficientdet \
        --object-name SM_TeslaM3_v2 \
        --texture experiments/phase1_random/final/texture_final.npy

    # Discover static objects
    python evaluation/transfer_eval.py --discover-objects tesla
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# Reuse utility functions from TextureEvaluator
from evaluation.TextureEvaluator import (
    load_texture, tile_and_upsample, draw_detections,
    filter_by_gt_bbox, filter_target_detections,
    render_confidence_graph, add_confidence_graphs,
    save_video, save_csv, compute_ap, compute_iou,
    mask_to_bbox, clean_segmentation_mask,
    orbit_camera, discover_objects, find_object_location, parse_location,
)

from evaluation.detectors import create_detector, SUPPORTED_DETECTORS


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Transfer evaluation: test adversarial texture on different detectors')
    parser.add_argument('--texture', type=str,
                        default='experiments/phase1_random/final/texture_final.npy',
                        help='Path to texture checkpoint (.npy or .pt)')
    parser.add_argument('--detector', type=str, default='yolov5s',
                        choices=SUPPORTED_DETECTORS,
                        help='Detector to evaluate (default: yolov5s)')
    parser.add_argument('--object-name', type=str, default=None,
                        help='CARLA static vehicle mesh name')
    parser.add_argument('--static-location', type=str, default=None,
                        help='World location as "x,y,z"')
    parser.add_argument('--discover-objects', type=str, default=None, metavar='KEYWORD',
                        help='List CARLA objects matching KEYWORD, then exit')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: evaluation/results/transfer/<detector>)')
    parser.add_argument('--yaw-step', type=int, default=1,
                        help='Yaw step in degrees (default: 1 = 360 views)')
    parser.add_argument('--pitch', type=float, default=-15)
    parser.add_argument('--distance', type=float, default=8)
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--skip-load-world', action='store_true',
                        help='Do not reload the map')
    parser.add_argument('--cloudiness', type=float, default=None)
    args = parser.parse_args()

    # Lazy import — only needed when actually connecting to CARLA
    import carla
    from scripts.CarlaHandler import CarlaHandler

    # --- Object discovery mode ---
    if args.discover_objects is not None:
        print(f"Connecting to CARLA to discover objects matching '{args.discover_objects}'...")
        handler = CarlaHandler(town=args.town, x_res=1024, y_res=1024,
                               skip_load_world=args.skip_load_world)
        matches = discover_objects(handler, args.discover_objects)
        print(f"\nFound {len(matches)} objects matching '{args.discover_objects}':")
        for name in matches:
            print(f"  {name}")
        if not matches:
            print("  (no matches)")
        return

    # --- Validate ---
    if args.object_name is None:
        parser.error("--object-name is required. Use --discover-objects to find it.")

    # Output directory
    if args.output_dir is None:
        output_dir = Path('evaluation/results/transfer') / args.detector
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"TRANSFER EVALUATION — {args.detector.upper()}")
    print("=" * 70)

    # Parse static location
    static_location = None
    if args.static_location is not None:
        static_location = parse_location(args.static_location)

    # Viewpoints
    viewpoints = [{'yaw': y, 'pitch': args.pitch, 'distance': args.distance}
                  for y in range(0, 360, args.yaw_step)]
    print(f"Viewpoints: {len(viewpoints)} (every {args.yaw_step} deg)")

    # Load texture
    print(f"\nLoading texture: {args.texture}")
    texture_chw = load_texture(args.texture)
    coarse_size = texture_chw.shape[1]
    texture_full = tile_and_upsample(texture_chw, tile_count=coarse_size, resolution=1024)
    texture_rgb = (np.clip(texture_full, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)

    # Connect to CARLA (rendering enabled — we need actual frames)
    print(f"\nConnecting to CARLA (skip_load_world=True)...")
    handler = CarlaHandler(town=args.town, x_res=1024, y_res=1024,
                           no_rendering=False, skip_load_world=True)
    if args.cloudiness is not None:
        handler.update_cloudiness(args.cloudiness)
        print(f"  Cloudiness set to {args.cloudiness}")

    # Find static object
    object_name_full = args.object_name
    if static_location is None:
        print(f"\nAuto-discovering location of '{args.object_name}'...")
        discovered_name, discovered_loc = find_object_location(handler, args.object_name)
        if discovered_loc is not None:
            static_location = discovered_loc
            object_name_full = discovered_name
        else:
            print(f"  ERROR: Could not find '{args.object_name}'.")
            matches = discover_objects(handler, args.object_name)
            for m in matches[:10]:
                print(f"    {m}")
            return

    print(f"  Object: {object_name_full}")
    print(f"  Location: ({static_location.x:.1f}, {static_location.y:.1f}, {static_location.z:.1f})")

    # Apply texture via streaming API
    print(f"\n  Applying texture via streaming API...")
    h, w = texture_rgb.shape[:2]
    try:
        tex = carla.TextureColor(w, h)
        for x in range(w):
            for y in range(h):
                r, g, b = int(texture_rgb[y, x, 0]), int(texture_rgb[y, x, 1]), int(texture_rgb[y, x, 2])
                tex.set(x, h - y - 1, carla.Color(r, g, b, 255))
        handler.world.apply_color_texture_to_object(
            object_name_full, carla.MaterialParameter.Diffuse, tex
        )
        print(f"  Texture applied ({w}x{h})")
    except RuntimeError as e:
        print(f"  Texture streaming API failed: {e}")
        print(f"  Assuming texture was applied in UE4 editor")
    handler.world_tick(30)
    time.sleep(0.5)

    # Initialize detector
    print(f"\nInitializing {args.detector}...")
    detector, det_size = create_detector(args.detector)
    device = next(detector.model.parameters()).device

    # --- Evaluation loop ---
    print(f"\n{'=' * 70}")
    print(f"RUNNING {args.detector.upper()} (det_size={det_size})")
    print(f"{'=' * 70}")

    results = []
    frames = []
    all_detections = []
    all_gt_boxes = []

    for i, vp in enumerate(viewpoints):
        yaw = vp['yaw']
        pitch = vp.get('pitch', -15)
        distance = vp.get('distance', 8)

        # Capture with retry
        for attempt in range(3):
            try:
                orbit_camera(handler, static_location, yaw, pitch, distance)
                handler.world_tick(20)
                time.sleep(0.3)
                img_bgr = handler.get_image()
                handler.world_tick(5)
                seg_mask = handler.get_car_segmentation_mask()
                seg_mask = clean_segmentation_mask(seg_mask)
                break
            except RuntimeError:
                if attempt < 2:
                    print(f"  View {i:2d} | yaw={yaw:3d} | timeout, retrying ({attempt+1}/3)...")
                    time.sleep(5)
                else:
                    raise

        gt_box = mask_to_bbox(seg_mask, image_size=1024, det_size=det_size)
        all_gt_boxes.append(gt_box)

        # Run detector
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_resized = F.interpolate(img_t, size=(det_size, det_size),
                                    mode='bilinear', align_corners=False)

        det_results = detector.detect_cars_with_boxes(img_resized, score_threshold=0.01)[0]
        det_results = filter_by_gt_bbox(det_results, gt_box, padding=20)
        all_detections.append(det_results)

        max_conf = det_results['scores'][0] if len(det_results['scores']) > 0 else 0.0

        if gt_box is not None and len(det_results['scores']) > 0:
            best_iou = compute_iou(det_results['boxes'][0], gt_box)
        else:
            best_iou = 0.0

        results.append({
            'viewpoint': i,
            'yaw': yaw,
            'pitch': pitch,
            'distance': distance,
            'detector': args.detector,
            'max_conf': float(max_conf),
            'num_detections': len(det_results['scores']),
            'iou': float(best_iou),
        })

        # Annotate frame
        frame = img_bgr.copy()
        draw_detections(frame, det_results, det_size)
        cv2.putText(frame, f"{args.detector} (yaw={yaw})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Max conf: {max_conf:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        frames.append(frame)
        cv2.imwrite(str(output_dir / f'yaw_{yaw:03d}.png'), frame)

        print(f"  View {i:2d} | yaw={yaw:3d} | conf={max_conf:.4f} | iou={best_iou:.3f}")

    # Compute AP@0.5
    ap50 = compute_ap(all_detections, all_gt_boxes, iou_threshold=0.5)

    # Save results
    save_csv(results, output_dir / 'detection_results.csv')
    confs = [r['max_conf'] for r in results]
    frames_with_graph = add_confidence_graphs(frames, confs)
    save_video(frames_with_graph, output_dir / 'evaluation.mp4', fps=args.fps)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS — {args.detector.upper()}")
    print(f"{'=' * 70}")
    print(f"  AP@0.5:              {ap50:.4f}")
    print(f"  Max confidence:      {max(confs):.4f}")
    print(f"  Mean confidence:     {np.mean(confs):.4f}")
    print(f"  Views conf > 0.5:   {sum(1 for c in confs if c > 0.5)}/{len(confs)}")
    print(f"  Views conf < 0.1:   {sum(1 for c in confs if c < 0.1)}/{len(confs)}")
    print(f"  Output:              {output_dir}")

    # Save summary
    summary = {
        'detector': args.detector,
        'texture': args.texture,
        'ap50': ap50,
        'max_conf': max(confs),
        'mean_conf': float(np.mean(confs)),
        'views_above_0.5': sum(1 for c in confs if c > 0.5),
        'views_below_0.1': sum(1 for c in confs if c < 0.1),
        'total_views': len(confs),
    }
    save_csv([summary], output_dir / 'summary.csv')


if __name__ == '__main__':
    main()
