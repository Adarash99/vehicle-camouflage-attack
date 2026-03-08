#!/usr/bin/env python3
"""
Texture Evaluator: Visualize adversarial textures on vehicle + detection evaluation

Loads an optimized adversarial texture checkpoint and evaluates it:
1. Applies texture to vehicle in CARLA (via texture streaming API if available,
   otherwise uses the neural renderer as fallback)
2. Orbits camera around vehicle at configurable viewpoints
3. Runs EfficientDet at each viewpoint and records detection confidence
4. Saves frames as video with detection confidence overlay
5. Outputs summary CSV with per-viewpoint detection results

Three rendering modes:
  - "neural": Use the U-Net3 neural renderer (same as training pipeline sees).
             This is the default and doesn't require the CARLA texture API.
  - "carla": Apply texture directly to a pre-placed static vehicle mesh via
             CARLA's texture streaming API. Requires a source-built CARLA with a
             static Tesla Model 3 placed in the UE4 editor.
             NOTE: Texture streaming does NOT work on spawned actor vehicles —
             only on static world objects (confirmed by probe_texture_api.py).
  - "both": Run both neural and CARLA evaluations.

CARLA mode usage:
    # Step 1: Discover the static object name (run once)
    python scripts/TextureEvaluator.py --discover-objects tesla

    # Step 2: Evaluate with the discovered object name + location
    python scripts/TextureEvaluator.py --mode carla \
        --object-name "SM_TeslaM3_v2" \
        --static-location "100.0,200.0,0.5" \
        --texture experiments/phase1_random/final/texture_final.npy

Neural mode usage:
    python scripts/TextureEvaluator.py --texture experiments/phase1_random/final/texture_final.npy
    python scripts/TextureEvaluator.py --texture experiments/phase1_random/checkpoints/texture_iter_0900.npy
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import torch
import torch.nn.functional as F

from scripts.CarlaHandler import CarlaHandler
from scripts.texture_applicator_pytorch import TextureApplicatorPyTorch
from attack.detector_pytorch import EfficientDetPyTorch


def orbit_camera(handler, location, yaw, pitch, distance):
    """Position the spectator camera at a viewpoint orbiting a fixed world location.

    Same math as CarlaHandler.set_camera_viewpoint but uses an arbitrary world
    point instead of requiring a spawned vehicle.

    Args:
        handler: CarlaHandler instance
        location: carla.Location of the point to orbit around
        yaw: Horizontal angle in degrees (0=forward along +X)
        pitch: Vertical angle in degrees (negative = looking down)
        distance: Distance from the target point in meters
    """
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    x_offset = math.cos(yaw_rad) * math.cos(pitch_rad) * distance
    y_offset = math.sin(yaw_rad) * math.cos(pitch_rad) * distance
    # Negate: pitch=-15 (look down) → camera should be ABOVE target
    z_offset = -math.sin(pitch_rad) * distance

    camera_pos = carla.Location(
        x=location.x - x_offset,
        y=location.y - y_offset,
        z=location.z + z_offset + 1.5,  # +1.5m for vehicle height
    )
    camera_rot = carla.Rotation(
        pitch=pitch,  # pitch=-15 → CARLA looks down
        yaw=yaw,  # camera is already behind target, face toward it
        roll=0,
    )
    handler.spectator.set_transform(carla.Transform(camera_pos, camera_rot))
    handler.world.tick()


def discover_objects(handler, keyword):
    """List all CARLA world object names matching a keyword.

    Args:
        handler: CarlaHandler instance
        keyword: Case-insensitive substring to search for

    Returns:
        list of matching object name strings
    """
    all_names = handler.world.get_names_of_all_objects()
    keyword_lower = keyword.lower()
    matches = [n for n in all_names if keyword_lower in n.lower()]
    return sorted(matches)


def find_object_location(handler, object_name):
    """Find the world location of a named static object using environment objects API.

    Tries multiple CARLA API methods to locate the object:
    1. get_environment_objects() for static meshes placed in the editor
    2. Fallback: search all actors

    Args:
        handler: CarlaHandler instance
        object_name: Name of the object (e.g. 'SM_TeslaM3_parked')

    Returns:
        (full_name, carla.Location) or (None, None) if not found
    """
    name_lower = object_name.lower()

    # Method 1: Search environment objects (works for static meshes in editor)
    try:
        env_objects = handler.world.get_environment_objects(carla.CityObjectLabel.Car)
        for obj in env_objects:
            if name_lower in obj.name.lower():
                loc = obj.transform.location
                print(f"  Found '{obj.name}' via environment objects at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
                return obj.name, loc
        # Also try Vehicles label
        env_objects = handler.world.get_environment_objects(carla.CityObjectLabel.Vehicles)
        for obj in env_objects:
            if name_lower in obj.name.lower():
                loc = obj.transform.location
                print(f"  Found '{obj.name}' via environment objects at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
                return obj.name, loc
    except Exception as e:
        print(f"  Environment objects API failed: {e}")

    # Method 2: Try all environment object types
    try:
        for label in [carla.CityObjectLabel.Any]:
            env_objects = handler.world.get_environment_objects(label)
            for obj in env_objects:
                if name_lower in obj.name.lower():
                    loc = obj.transform.location
                    print(f"  Found '{obj.name}' via env objects (Any) at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
                    return obj.name, loc
    except Exception as e:
        print(f"  Environment objects (Any) API failed: {e}")

    # Method 3: Search spawned actors
    try:
        for actor in handler.world.get_actors():
            if hasattr(actor, 'type_id') and name_lower in actor.type_id.lower():
                loc = actor.get_location()
                print(f"  Found actor '{actor.type_id}' at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
                return actor.type_id, loc
    except Exception as e:
        print(f"  Actor search failed: {e}")

    return None, None


def load_texture(texture_path, coarse_size=16):
    """Load texture from .npy or .pt file.

    Returns:
        numpy array [3, coarse_size, coarse_size] float32 in [0, 1]
    """
    path = Path(texture_path)
    if path.suffix == '.npy':
        texture = np.load(str(path))
    elif path.suffix == '.pt':
        texture = torch.load(str(path), map_location='cpu').numpy()
    else:
        raise ValueError(f"Unsupported texture format: {path.suffix}")

    if texture.ndim == 2:
        raise ValueError(f"Expected 3D texture [C, H, W], got {texture.shape}")
    print(f"  Loaded texture: {texture.shape}, range [{texture.min():.3f}, {texture.max():.3f}]")
    return texture.astype(np.float32)


def tile_and_upsample(texture_chw, tile_count=None, resolution=1024):
    """Tile a coarse texture and upsample to full resolution.

    Args:
        texture_chw: [3, H, W] float32
        tile_count: How many times to tile (default: same as texture size)
        resolution: Target resolution

    Returns:
        [3, resolution, resolution] float32
    """
    if tile_count is None:
        tile_count = texture_chw.shape[1]
    t = torch.from_numpy(texture_chw).unsqueeze(0)  # [1, 3, H, W]
    tiled = t.repeat(1, 1, tile_count, tile_count)
    upsampled = F.interpolate(tiled, size=(resolution, resolution), mode='nearest')
    return upsampled.squeeze(0).numpy()  # [3, resolution, resolution]


def draw_detections(image_bgr, detections, det_input_size=512):
    """Draw detection bounding boxes and scores on an image.

    Args:
        image_bgr: BGR uint8 image to draw on (modified in place)
        detections: dict with 'boxes' [N, 4] and 'scores' [N]
        det_input_size: Size the detector used (for coordinate scaling)
    """
    h, w = image_bgr.shape[:2]
    scale_x = w / det_input_size
    scale_y = h / det_input_size

    if len(detections['scores']) == 0:
        return
    # Draw only the highest-confidence detection
    score = detections['scores'][0]
    x1, y1, x2, y2 = detections['boxes'][0]
    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

    color = (0, 0, 255) if score > 0.5 else (0, 255, 0)
    thickness = 3 if score > 0.5 else 2
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
    label = f"{score:.3f}"
    cv2.putText(image_bgr, label, (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def filter_target_detections(detections, det_input_size=512, center_fraction=0.7):
    """Filter detections to only keep those near the center of the frame.

    Fallback filter when no GT bounding box is available. Keeps detections
    whose box center falls within the center region of the image.

    Args:
        detections: dict with 'boxes' [N, 4] (x1,y1,x2,y2) and 'scores' [N]
        det_input_size: Detector input resolution (512)
        center_fraction: Fraction of image considered "center" (0.7 = center 70%)

    Returns:
        Filtered detections dict (same format, background detections removed)
    """
    if len(detections['scores']) == 0:
        return detections

    boxes = detections['boxes']
    scores = detections['scores']

    margin = (1.0 - center_fraction) / 2.0 * det_input_size
    cx_min, cx_max = margin, det_input_size - margin
    cy_min, cy_max = margin, det_input_size - margin

    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    mask = (box_cx >= cx_min) & (box_cx <= cx_max) & \
           (box_cy >= cy_min) & (box_cy <= cy_max)

    if mask.sum() == 0:
        return {'boxes': np.empty((0, 4)), 'scores': np.array([])}

    return {'boxes': boxes[mask], 'scores': scores[mask]}


def filter_by_gt_bbox(detections, gt_box, padding=20):
    """Filter detections to only keep those overlapping with the GT bounding box.

    Uses the segmentation-derived GT bbox (expanded by padding) to reject
    background false positives. Only detections whose center falls within
    the padded GT region are kept.

    Args:
        detections: dict with 'boxes' [N, 4] (x1,y1,x2,y2) and 'scores' [N]
        gt_box: [x1, y1, x2, y2] in detector coordinates, or None
        padding: Pixels to expand GT box by on each side (default 20)

    Returns:
        Filtered detections dict
    """
    if len(detections['scores']) == 0 or gt_box is None:
        return detections

    boxes = detections['boxes']
    scores = detections['scores']

    # Expand GT box by padding
    gx1 = gt_box[0] - padding
    gy1 = gt_box[1] - padding
    gx2 = gt_box[2] + padding
    gy2 = gt_box[3] + padding

    # Keep detections whose center is within the padded GT box
    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    mask = (box_cx >= gx1) & (box_cx <= gx2) & \
           (box_cy >= gy1) & (box_cy <= gy2)

    if mask.sum() == 0:
        return {'boxes': np.empty((0, 4)), 'scores': np.array([])}

    return {'boxes': boxes[mask], 'scores': scores[mask]}


def evaluate_neural_renderer(args, handler, detector, renderer, texture_chw, viewpoints, output_dir):
    """Evaluate using the neural renderer (same view as training pipeline).

    For each viewpoint:
    1. Capture reference image + mask from CARLA
    2. Apply tiled texture via neural renderer
    3. Run EfficientDet on the rendered image
    4. Record confidence and save annotated frame
    """
    device = next(detector.model.parameters()).device
    resolution = 1024
    det_size = 512
    coarse_size = texture_chw.shape[1]

    # Tile and upsample texture
    texture_full = tile_and_upsample(texture_chw, tile_count=coarse_size, resolution=resolution)
    texture_full_t = torch.from_numpy(texture_full).float().to(device)  # [3, 1024, 1024]

    results = []
    frames = []

    for i, vp in enumerate(viewpoints):
        yaw = vp['yaw']
        pitch = vp.get('pitch', -15)
        distance = vp.get('distance', 8)

        # Position camera
        handler.set_camera_viewpoint(yaw=yaw, pitch=pitch, distance=distance)
        handler.world_tick(30)
        time.sleep(0.5)

        # Capture reference image and mask
        img_bgr = handler.get_image()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        handler.world_tick(20)
        car_mask = handler.get_car_segmentation_mask()  # [H, W]

        # Convert to tensors [1, C, H, W]
        x_ref_t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_t = torch.from_numpy(car_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        tex_t = (texture_full_t.unsqueeze(0) * mask_t)

        # Neural render
        with torch.no_grad():
            rendered_t = renderer.apply_differentiable(x_ref_t, tex_t, mask_t)

        # Run detector
        rendered_resized = F.interpolate(rendered_t, size=(det_size, det_size),
                                         mode='bilinear', align_corners=False)
        det_results = detector.detect_cars_with_boxes(rendered_resized, score_threshold=0.01)[0]
        det_results = filter_target_detections(det_results, det_size)

        # Also run detector on the reference (no texture) for comparison
        ref_resized = F.interpolate(x_ref_t, size=(det_size, det_size),
                                    mode='bilinear', align_corners=False)
        ref_det = detector.detect_cars_with_boxes(ref_resized, score_threshold=0.01)[0]
        ref_det = filter_target_detections(ref_det, det_size)

        # Get max confidence
        max_conf_adv = det_results['scores'][0] if len(det_results['scores']) > 0 else 0.0
        max_conf_ref = ref_det['scores'][0] if len(ref_det['scores']) > 0 else 0.0

        results.append({
            'viewpoint': i,
            'yaw': yaw,
            'pitch': pitch,
            'distance': distance,
            'max_conf_adversarial': float(max_conf_adv),
            'max_conf_reference': float(max_conf_ref),
            'num_detections_adv': len(det_results['scores']),
            'num_detections_ref': len(ref_det['scores']),
        })

        # Build annotated frame for video
        rendered_np = rendered_t[0].detach().cpu().permute(1, 2, 0).numpy()
        rendered_bgr = (rendered_np[:, :, ::-1] * 255).astype(np.uint8)

        ref_bgr = img_bgr.copy()

        # Draw detections on both
        draw_detections(rendered_bgr, det_results, det_size)
        draw_detections(ref_bgr, ref_det, det_size)

        # Add labels
        cv2.putText(rendered_bgr, f"Adversarial (yaw={yaw})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rendered_bgr, f"Max conf: {max_conf_adv:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(ref_bgr, f"Reference (yaw={yaw})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ref_bgr, f"Max conf: {max_conf_ref:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Side-by-side composite
        composite = np.hstack([ref_bgr, rendered_bgr])
        frames.append(composite)

        # Save individual frame
        cv2.imwrite(str(output_dir / f'frame_yaw_{yaw:03d}.png'), composite)

        print(f"  View {i:2d} | yaw={yaw:3d} | ref_conf={max_conf_ref:.4f} | adv_conf={max_conf_adv:.4f}")

    return results, frames


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clean_segmentation_mask(seg_mask):
    """Keep only the largest connected component in the segmentation mask.

    Removes small stray patches (e.g. distant background cars) that would
    inflate the bounding box.

    Args:
        seg_mask: [H, W] float32 mask (1=car, 0=background)

    Returns:
        Cleaned mask with only the largest connected component.
    """
    mask_uint8 = (seg_mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return seg_mask

    # Find the largest component (skip label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    cleaned = np.zeros_like(seg_mask)
    cleaned[labels == largest_label] = 1.0
    return cleaned


def mask_to_bbox(seg_mask, image_size=1024, det_size=512):
    """Convert a binary segmentation mask to a bounding box in detector coordinates.

    Args:
        seg_mask: [H, W] float32, 1.0 = car pixel
        image_size: Original image resolution
        det_size: Detector input resolution

    Returns:
        [x1, y1, x2, y2] in detector coordinates, or None if no car pixels
    """
    if seg_mask.sum() == 0:
        return None
    ys, xs = np.where(seg_mask > 0.5)
    scale = det_size / image_size
    return np.array([xs.min() * scale, ys.min() * scale,
                     xs.max() * scale, ys.max() * scale], dtype=np.float32)


def compute_ap(all_detections, all_gt_boxes, iou_threshold=0.5):
    """Compute Average Precision at a given IoU threshold.

    Args:
        all_detections: list of dicts per frame, each with 'boxes' [N,4] and 'scores' [N]
        all_gt_boxes: list of GT boxes per frame ([x1,y1,x2,y2] or None if no GT)
        iou_threshold: IoU threshold for a true positive (default 0.5)

    Returns:
        AP value (float)
    """
    # Collect all detections across frames with their frame index
    det_list = []  # (score, frame_idx, box)
    for frame_idx, det in enumerate(all_detections):
        for j in range(len(det['scores'])):
            det_list.append((float(det['scores'][j]), frame_idx, det['boxes'][j]))

    # Sort by confidence descending
    det_list.sort(key=lambda x: -x[0])

    n_gt = sum(1 for gt in all_gt_boxes if gt is not None)
    if n_gt == 0:
        return 0.0

    # Track which GT boxes have been matched
    gt_matched = [False] * len(all_gt_boxes)

    tp = np.zeros(len(det_list))
    fp = np.zeros(len(det_list))

    for i, (score, frame_idx, box) in enumerate(det_list):
        gt_box = all_gt_boxes[frame_idx]
        if gt_box is not None and not gt_matched[frame_idx]:
            iou = compute_iou(box, gt_box)
            if iou >= iou_threshold:
                tp[i] = 1
                gt_matched[frame_idx] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Cumulative sums
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # AP using all-point interpolation (PASCAL VOC style)
    # Prepend (0, 1) and append (1, 0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute area under curve
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return float(ap)


def evaluate_carla_texture(args, handler, detector, texture_chw, viewpoints,
                           output_dir, object_name, static_location):
    """Evaluate by applying texture to a static vehicle mesh in CARLA.

    The texture streaming API works on static world objects (placed in UE4 editor)
    but NOT on spawned actor vehicles. This function:
    1. Applies the adversarial texture to the named static object
    2. Orbits the camera around the static object's known location
    3. Runs EfficientDet at each viewpoint
    4. Computes AP@0.5 using segmentation masks as ground truth

    Args:
        args: CLI arguments
        handler: CarlaHandler instance (rendering must be enabled)
        detector: EfficientDetPyTorch instance
        texture_chw: [3, H, W] float32 texture in [0, 1]
        viewpoints: list of dicts with yaw/pitch/distance
        output_dir: Path for output frames/video/CSV
        object_name: CARLA object name for the static vehicle mesh
        static_location: carla.Location of the static vehicle in the world
    """
    det_size = 512
    device = next(detector.model.parameters()).device

    # Try to apply adversarial texture via texture streaming API
    # Falls back gracefully if the object is a skeletal mesh (actor) where
    # the API doesn't work — assumes texture was applied in UE4 editor
    coarse_size = texture_chw.shape[1]
    texture_full = tile_and_upsample(texture_chw, tile_count=coarse_size, resolution=1024)
    texture_rgb = (np.clip(texture_full, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)

    print(f"  Attempting to apply texture to '{object_name}' via streaming API...")
    h, w = texture_rgb.shape[:2]
    try:
        tex = carla.TextureColor(w, h)
        for x in range(w):
            for y in range(h):
                r, g, b = int(texture_rgb[y, x, 0]), int(texture_rgb[y, x, 1]), int(texture_rgb[y, x, 2])
                tex.set(x, h - y - 1, carla.Color(r, g, b, 255))
        handler.world.apply_color_texture_to_object(
            object_name, carla.MaterialParameter.Diffuse, tex
        )
        print(f"  Texture applied via API ({w}x{h})")
    except RuntimeError as e:
        print(f"  Texture streaming API failed: {e}")
        print(f"  Assuming texture was applied in UE4 editor — proceeding with evaluation")
    handler.world_tick(30)
    time.sleep(0.5)

    results = []
    frames = []
    all_detections = []
    all_gt_boxes = []

    for i, vp in enumerate(viewpoints):
        yaw = vp['yaw']
        pitch = vp.get('pitch', -15)
        distance = vp.get('distance', 8)

        # Retry loop for CARLA timeout resilience
        for attempt in range(3):
            try:
                orbit_camera(handler, static_location, yaw, pitch, distance)
                handler.world_tick(20)
                time.sleep(0.3)

                # Capture the actual CARLA render with texture applied
                img_bgr = handler.get_image()

                # Get segmentation mask for GT bounding box
                handler.world_tick(5)
                seg_mask = handler.get_car_segmentation_mask()  # [H, W] float32
                seg_mask = clean_segmentation_mask(seg_mask)
                break  # success
            except RuntimeError as e:
                if attempt < 2:
                    print(f"  View {i:2d} | yaw={yaw:3d} | CARLA timeout, retrying ({attempt+1}/3)...")
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

        # Compute per-frame IoU for logging
        if gt_box is not None and len(det_results['scores']) > 0:
            best_iou = compute_iou(det_results['boxes'][0], gt_box)
        else:
            best_iou = 0.0

        results.append({
            'viewpoint': i,
            'yaw': yaw,
            'pitch': pitch,
            'distance': distance,
            'max_conf_carla': float(max_conf),
            'num_detections': len(det_results['scores']),
            'iou': float(best_iou),
        })

        # Annotate frame
        frame = img_bgr.copy()
        draw_detections(frame, det_results, det_size)
        cv2.putText(frame, f"CARLA Render (yaw={yaw})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Max conf: {max_conf:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        frames.append(frame)
        cv2.imwrite(str(output_dir / f'carla_yaw_{yaw:03d}.png'), frame)

        print(f"  View {i:2d} | yaw={yaw:3d} | conf={max_conf:.4f} | iou={best_iou:.3f}")

    # Compute AP@0.5
    ap50 = compute_ap(all_detections, all_gt_boxes, iou_threshold=0.5)
    print(f"\n  AP@0.5 = {ap50:.4f}")

    return results, frames, ap50


def render_confidence_graph(all_confs, current_idx, width, height=200, threshold=0.5):
    """Render a confidence score line graph as a BGR image.

    Shows the full confidence curve across all viewpoints with a vertical
    marker at the current frame position, similar to DTA evaluation videos.

    Args:
        all_confs: list of confidence scores for all frames
        current_idx: index of the current frame (vertical marker)
        width: width of the graph image (matches video frame width)
        height: height of the graph image
        threshold: confidence threshold for horizontal line (default 0.5)

    Returns:
        BGR uint8 image [height, width, 3]
    """
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    graph[:] = (30, 30, 30)  # dark background

    n = len(all_confs)
    if n == 0:
        return graph

    # Plot area margins
    margin_left = 60
    margin_right = 20
    margin_top = 28
    margin_bottom = 30
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Red/green background shading (above/below threshold)
    thresh_y = int(margin_top + plot_h * (1.0 - threshold))
    overlay = graph.copy()
    cv2.rectangle(overlay, (margin_left, margin_top),
                  (margin_left + plot_w, thresh_y), (0, 0, 50), -1)
    cv2.rectangle(overlay, (margin_left, thresh_y),
                  (margin_left + plot_w, margin_top + plot_h), (0, 35, 0), -1)
    cv2.addWeighted(overlay, 0.6, graph, 0.4, 0, graph)

    # Grid lines and Y-axis labels
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(margin_top + plot_h * (1.0 - val))
        cv2.line(graph, (margin_left, y), (margin_left + plot_w, y),
                 (60, 60, 60), 1)
        cv2.putText(graph, f"{val:.2f}", (3, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Threshold line (dashed-style red)
    cv2.line(graph, (margin_left, thresh_y), (margin_left + plot_w, thresh_y),
             (0, 0, 200), 2, cv2.LINE_AA)

    # Compute point positions
    points = []
    for i, conf in enumerate(all_confs):
        x = int(margin_left + (i / max(n - 1, 1)) * plot_w)
        y = int(margin_top + plot_h * (1.0 - min(max(conf, 0.0), 1.0)))
        points.append((x, y))

    # Draw confidence line up to current frame only (progressive reveal)
    draw_up_to = min(current_idx, n - 1) if current_idx >= 0 else -1
    for i in range(draw_up_to):
        cv2.line(graph, points[i], points[i + 1], (255, 160, 40), 2, cv2.LINE_AA)

    # Current frame marker dot
    if 0 <= draw_up_to < n:
        cv2.circle(graph, points[draw_up_to], 5, (0, 255, 255), -1, cv2.LINE_AA)

    # X-axis labels (degree markers)
    max_yaw = 360
    for deg in range(0, max_yaw + 1, 60):
        frac = deg / max_yaw
        x = int(margin_left + frac * plot_w)
        cv2.putText(graph, f"{deg}", (x - 10, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # Title bar with stats (up to current frame)
    confs_so_far = all_confs[:current_idx + 1] if current_idx >= 0 else []
    avg_conf = np.mean(confs_so_far) if confs_so_far else 0.0
    n_so_far = max(len(confs_so_far), 1)
    det_rate = sum(1 for c in confs_so_far if c > threshold) / n_so_far * 100
    cv2.putText(graph, "Car Confidence Score", (margin_left, margin_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    stats_text = f"Mean: {avg_conf:.3f} | Det rate (>{threshold}): {det_rate:.1f}%"
    cv2.putText(graph, stats_text, (width // 2, margin_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Plot border
    cv2.rectangle(graph, (margin_left, margin_top),
                  (margin_left + plot_w, margin_top + plot_h), (100, 100, 100), 1)

    return graph


def add_confidence_graphs(frames, confidences):
    """Append a confidence graph panel to the bottom of each video frame.

    Args:
        frames: list of BGR images
        confidences: list of float confidence values (one per frame)

    Returns:
        list of BGR images with graph composited at the bottom
    """
    if not frames or not confidences:
        return frames

    composites = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        graph_h = max(180, h // 5)
        graph = render_confidence_graph(confidences, i, w, graph_h)
        composites.append(np.vstack([frame, graph]))
    return composites


def save_video(frames, output_path, fps=2):
    """Save list of BGR frames as MP4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  Video saved: {output_path} ({len(frames)} frames, {fps} fps)")


def save_csv(results, output_path):
    """Save results list of dicts to CSV."""
    if not results:
        return
    fieldnames = results[0].keys()
    with open(str(output_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {output_path}")


def parse_location(loc_str):
    """Parse 'x,y,z' string into a carla.Location."""
    parts = [float(p.strip()) for p in loc_str.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 'x,y,z' format, got '{loc_str}' ({len(parts)} values)")
    return carla.Location(x=parts[0], y=parts[1], z=parts[2])


def main():
    parser = argparse.ArgumentParser(description='Texture Evaluator')
    parser.add_argument('--texture', type=str,
                        default='experiments/phase1_random/final/texture_final.npy',
                        help='Path to texture checkpoint (.npy or .pt)')
    parser.add_argument('--mode', choices=['neural', 'carla', 'both'], default='neural',
                        help='Rendering mode: neural (U-Net3), carla (texture API), or both')
    parser.add_argument('--object-name', type=str, default=None,
                        help='CARLA object name for the static vehicle mesh '
                             '(placed in UE4 editor, found via --discover-objects)')
    parser.add_argument('--static-location', type=str, default=None,
                        help='World location of static vehicle as "x,y,z" '
                             '(required for carla mode)')
    parser.add_argument('--discover-objects', type=str, default=None, metavar='KEYWORD',
                        help='List all CARLA world objects matching KEYWORD, then exit. '
                             'Use this to find the name of your static vehicle mesh.')
    parser.add_argument('--output-dir', type=str, default='evaluation/results/',
                        help='Output directory')
    parser.add_argument('--yaw-step', type=int, default=1,
                        help='Yaw angle step in degrees (default: 1 = 360 viewpoints)')
    parser.add_argument('--pitch', type=float, default=-15,
                        help='Camera pitch angle')
    parser.add_argument('--distance', type=float, default=8,
                        help='Camera distance from vehicle')
    parser.add_argument('--town', type=str, default='Town01',
                        help='CARLA town to use')
    parser.add_argument('--fps', type=int, default=20,
                        help='Video frame rate')
    parser.add_argument('--skip-load-world', action='store_true',
                        help='Do not reload the map (use when connecting to UE4 editor)')
    parser.add_argument('--renderer-path', type=str,
                        default='models/unet3/trained/best_model.pt',
                        help='Path to neural renderer weights')
    parser.add_argument('--cloudiness', type=float, default=None,
                        help='Override cloudiness (0-100)')
    args = parser.parse_args()

    # --- Object discovery mode (run and exit) ---
    if args.discover_objects is not None:
        print(f"Connecting to CARLA to discover objects matching '{args.discover_objects}'...")
        handler = CarlaHandler(town=args.town, x_res=1024, y_res=1024,
                               skip_load_world=args.skip_load_world)
        matches = discover_objects(handler, args.discover_objects)
        print(f"\nFound {len(matches)} objects matching '{args.discover_objects}':")
        for name in matches:
            print(f"  {name}")
        if not matches:
            print("  (no matches — try a broader keyword)")
        return

    print("=" * 70)
    print("TEXTURE EVALUATOR")
    print("=" * 70)
    print()

    # Validate args for carla mode
    if args.mode in ('carla', 'both'):
        if args.object_name is None:
            parser.error("--object-name is required for carla mode. "
                         "Use --discover-objects to find the right name.")

    # Parse static location if provided
    static_location = None
    if args.static_location is not None:
        static_location = parse_location(args.static_location)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build viewpoint list
    viewpoints = [{'yaw': y, 'pitch': args.pitch, 'distance': args.distance}
                  for y in range(0, 360, args.yaw_step)]
    print(f"Viewpoints: {len(viewpoints)} (every {args.yaw_step} degrees)")

    # Load texture
    print(f"\nLoading texture: {args.texture}")
    texture_chw = load_texture(args.texture)

    # Determine if we need rendering enabled
    # Neural mode uses no_rendering=True for performance (only needs segmentation data)
    # CARLA mode needs rendering enabled to capture actual rendered frames
    need_rendering = args.mode in ('carla', 'both')
    no_rendering = not need_rendering

    # Auto-enable skip_load_world for carla mode (editor workflow)
    skip_load = args.skip_load_world or args.mode in ('carla', 'both')

    # Connect to CARLA
    print(f"\nConnecting to CARLA (no_rendering={no_rendering}, skip_load_world={skip_load})...")
    handler = CarlaHandler(town=args.town, x_res=1024, y_res=1024,
                           no_rendering=no_rendering, skip_load_world=skip_load)

    # Override weather if requested
    if args.cloudiness is not None:
        handler.update_cloudiness(args.cloudiness)
        print(f"  Cloudiness set to {args.cloudiness}")

    # Auto-discover static mesh location if not provided
    object_name_full = args.object_name
    if args.mode in ('carla', 'both') and static_location is None:
        print(f"\nAuto-discovering location of '{args.object_name}'...")
        discovered_name, discovered_loc = find_object_location(handler, args.object_name)
        if discovered_loc is not None:
            static_location = discovered_loc
            object_name_full = discovered_name
        else:
            print(f"  ERROR: Could not find '{args.object_name}' in the world.")
            print(f"  Available matching objects:")
            matches = discover_objects(handler, args.object_name)
            for m in matches[:10]:
                print(f"    {m}")
            print(f"\n  Please provide --static-location 'x,y,z' manually.")
            return

    # Spawn a vehicle for neural mode (needs reference images + segmentation mask)
    # CARLA mode uses a static object placed in UE4 editor — no spawning needed
    need_vehicle = args.mode in ('neural', 'both')
    if need_vehicle:
        handler.spawn_vehicle('vehicle.tesla.model3', color=(124, 124, 124))
        handler.world_tick(50)
        time.sleep(1.0)
        print("  Vehicle spawned (for neural renderer evaluation)")
    else:
        handler.world_tick(30)
        time.sleep(0.5)
        print("  Connected (static object mode — no vehicle spawned)")

    # Initialize detector
    print("\nInitializing EfficientDet-D0...")
    detector = EfficientDetPyTorch()

    # --- Neural renderer evaluation ---
    if args.mode in ('neural', 'both'):
        print("\n" + "=" * 70)
        print("NEURAL RENDERER EVALUATION")
        print("=" * 70)

        renderer = TextureApplicatorPyTorch(model_path=args.renderer_path)

        neural_dir = output_dir / 'neural'
        neural_dir.mkdir(exist_ok=True)

        neural_results, neural_frames = evaluate_neural_renderer(
            args, handler, detector, renderer, texture_chw, viewpoints, neural_dir
        )

        # Save results
        save_csv(neural_results, neural_dir / 'detection_results.csv')
        adv_confs = [r['max_conf_adversarial'] for r in neural_results]
        neural_frames = add_confidence_graphs(neural_frames, adv_confs)
        save_video(neural_frames, neural_dir / 'evaluation.mp4', fps=args.fps)

        # Print summary
        ref_confs = [r['max_conf_reference'] for r in neural_results]
        print(f"\n  Neural Renderer Summary:")
        print(f"    Reference max conf:    {max(ref_confs):.4f} (mean {np.mean(ref_confs):.4f})")
        print(f"    Adversarial max conf:  {max(adv_confs):.4f} (mean {np.mean(adv_confs):.4f})")
        print(f"    Views with conf > 0.5: {sum(1 for c in adv_confs if c > 0.5)}/{len(adv_confs)}")
        print(f"    Views with conf < 0.1: {sum(1 for c in adv_confs if c < 0.1)}/{len(adv_confs)}")

    # --- CARLA texture API evaluation ---
    if args.mode in ('carla', 'both'):
        print("\n" + "=" * 70)
        print("CARLA TEXTURE API EVALUATION (static object)")
        print("=" * 70)
        print(f"  Object name: {object_name_full}")
        print(f"  Location: ({static_location.x}, {static_location.y}, {static_location.z})")

        if not need_rendering:
            # If we started with no_rendering=True, we need to switch
            settings = handler.world.get_settings()
            settings.no_rendering_mode = False
            handler.world.apply_settings(settings)
            handler.world_tick(30)

        carla_dir = output_dir if args.mode == 'carla' else output_dir / 'carla'
        carla_dir.mkdir(parents=True, exist_ok=True)

        carla_results, carla_frames, ap50 = evaluate_carla_texture(
            args, handler, detector, texture_chw, viewpoints, carla_dir,
            object_name_full, static_location
        )

        if carla_results is not None:
            save_csv(carla_results, carla_dir / 'detection_results.csv')
            carla_confs = [r['max_conf_carla'] for r in carla_results]
            carla_frames = add_confidence_graphs(carla_frames, carla_confs)
            save_video(carla_frames, carla_dir / 'evaluation.mp4', fps=args.fps)

            # Save summary with AP@0.5
            summary = {
                'ap50': ap50,
                'max_conf': max(carla_confs),
                'mean_conf': float(np.mean(carla_confs)),
                'views_above_0.5': sum(1 for c in carla_confs if c > 0.5),
                'views_below_0.1': sum(1 for c in carla_confs if c < 0.1),
                'total_views': len(carla_confs),
            }
            with open(str(carla_dir / 'summary.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary.keys())
                writer.writeheader()
                writer.writerow(summary)
            print(f"  Summary saved: {carla_dir / 'summary.csv'}")

            print(f"\n  CARLA Texture API Summary:")
            print(f"    AP@0.5:    {ap50:.4f}")
            print(f"    Max conf:  {max(carla_confs):.4f} (mean {np.mean(carla_confs):.4f})")
            print(f"    Views with conf > 0.5: {sum(1 for c in carla_confs if c > 0.5)}/{len(carla_confs)}")
            print(f"    Views with conf < 0.1: {sum(1 for c in carla_confs if c < 0.1)}/{len(carla_confs)}")
        else:
            print("  CARLA texture API failed — skipping.")

    # Cleanup
    if need_vehicle:
        handler.destroy_all_vehicles()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
