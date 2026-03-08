#!/usr/bin/env python3
"""
Visualize U-Net3 Renderer Output (L1 + Perceptual, Multi-Dataset trained model)

Produces three output images:
  1. Solid color grid (5 rows from dataset_8k_revised)
  2. Multicolor grid (5 rows from dataset_multicolor)
  3. Random texture grid (5 rows with random block textures)

Each row: Reference | Texture | Mask | Ground Truth | Generated

Usage:
    python models/unet3/test_scripts/visualize_unet3.py
    python models/unet3/test_scripts/visualize_unet3.py --model models/unet3/trained/best_model.pt
"""

import sys
import os
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts'))

import torch
import numpy as np
import cv2
from pathlib import Path

from texture_applicator_pytorch import TextureApplicatorPyTorch


def load_sample(dataset_path, sample_id):
    """Load a single dataset sample by ID."""
    base = Path(dataset_path)

    ref_path = base / 'reference' / f'{sample_id}.png'
    tex_path = base / 'texture' / f'{sample_id}.png'
    mask_path = base / 'mask' / f'{sample_id}.png'
    rendered_path = base / 'rendered' / f'{sample_id}.png'

    for p in [ref_path, tex_path, mask_path, rendered_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    ref = cv2.imread(str(ref_path))
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    tex = cv2.imread(str(tex_path))
    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    rendered = cv2.imread(str(rendered_path))
    rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

    return ref, tex, mask, rendered


def generate_random_texture(mask, coarse_size=128, seed=None):
    """Generate a random block texture masked to the car shape."""
    if seed is not None:
        np.random.seed(seed)

    h, w = mask.shape[:2]
    coarse = np.random.rand(coarse_size, coarse_size, 3).astype(np.float32)
    scale_h = h // coarse_size
    scale_w = w // coarse_size
    texture_full = np.kron(coarse, np.ones((scale_h, scale_w, 1))).astype(np.float32)

    mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    texture_masked = texture_full * mask_f

    return (texture_masked * 255).astype(np.uint8)


def run_renderer(model, ref, tex, mask, device):
    """Run renderer on a single sample. All inputs are numpy uint8."""
    ref_t = torch.from_numpy(ref).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tex_t = torch.from_numpy(tex).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0

    ref_t = ref_t.to(device)
    tex_t = tex_t.to(device)
    mask_t = mask_t.to(device)

    with torch.no_grad():
        output = model.forward_from_components(ref_t, tex_t, mask_t)

    output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    return output_np


def make_grid(rows, col_labels, img_size=1024, padding=4, label_height=40):
    """Create a grid image from rows of images at full resolution."""
    ncols = len(col_labels)
    nrows = len(rows)

    cell_w = img_size + padding
    cell_h = img_size + padding
    grid_w = ncols * cell_w + padding
    grid_h = label_height + nrows * cell_h + padding

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    for c, label in enumerate(col_labels):
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        x = padding + c * cell_w + (img_size - text_size[0]) // 2
        cv2.putText(grid, label, (x, label_height - 10), font, font_scale, (220, 220, 220), 2, cv2.LINE_AA)

    for r, row in enumerate(rows):
        y_off = label_height + r * cell_h + padding

        for c, img in enumerate(row['images']):
            x_off = padding + c * cell_w
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            grid[y_off:y_off + img_size, x_off:x_off + img_size] = img

        sid = str(row['sample_id'])
        cv2.putText(grid, sid, (padding + 4, y_off + 25), font, 0.7, (180, 180, 80), 1, cv2.LINE_AA)

    return grid


def get_sample_ids(dataset_path, count=5, seed=42):
    """Get random sample IDs from a dataset."""
    ref_dir = Path(dataset_path) / 'reference'
    all_ids = sorted([int(p.stem) for p in ref_dir.glob('*.png')])
    rng = np.random.RandomState(seed)
    selected = rng.choice(all_ids, size=min(count, len(all_ids)), replace=False)
    return sorted(selected.tolist())


def main():
    parser = argparse.ArgumentParser(description='Visualize U-Net3 renderer output')
    parser.add_argument('--model', default='models/unet3/trained/best_model.pt', help='Path to trained model')
    parser.add_argument('--solid-dataset', default='dataset_8k_revised/val', help='Solid color dataset path')
    parser.add_argument('--multi-dataset', default='dataset_multicolor/val', help='Multicolor dataset path')
    parser.add_argument('--rows', type=int, default=5, help='Number of rows per grid')
    parser.add_argument('--output-dir', default=os.path.join(_SCRIPT_DIR, 'output'), help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"Loading model from {args.model}...")
    applicator = TextureApplicatorPyTorch(model_path=args.model, device=device)
    model = applicator.model

    col_labels = ['Reference', 'Texture', 'Mask', 'Ground Truth', 'Generated']

    # --- Grid 1: Solid color samples ---
    print(f"\n--- Solid color samples from {args.solid_dataset} ---")
    solid_ids = get_sample_ids(args.solid_dataset, count=args.rows, seed=args.seed)
    solid_rows = []
    for sid in solid_ids:
        try:
            ref, tex, mask, gt = load_sample(args.solid_dataset, sid)
            gen = run_renderer(model, ref, tex, mask, device)
            solid_rows.append({'images': [ref, tex, mask, gt, gen], 'sample_id': sid})
            print(f"  Sample {sid}: OK")
        except FileNotFoundError as e:
            print(f"  Sample {sid}: SKIPPED ({e})")

    if solid_rows:
        grid = make_grid(solid_rows, col_labels)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, 'unet3_solid.png')
        cv2.imwrite(out_path, grid_bgr)
        print(f"Saved {out_path}")

    # --- Grid 2: Multicolor samples ---
    print(f"\n--- Multicolor samples from {args.multi_dataset} ---")
    multi_ids = get_sample_ids(args.multi_dataset, count=args.rows, seed=args.seed + 1)
    multi_rows = []
    for sid in multi_ids:
        try:
            ref, tex, mask, gt = load_sample(args.multi_dataset, sid)
            gen = run_renderer(model, ref, tex, mask, device)
            multi_rows.append({'images': [ref, tex, mask, gt, gen], 'sample_id': sid})
            print(f"  Sample {sid}: OK")
        except FileNotFoundError as e:
            print(f"  Sample {sid}: SKIPPED ({e})")

    if multi_rows:
        grid = make_grid(multi_rows, col_labels)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, 'unet3_multicolor.png')
        cv2.imwrite(out_path, grid_bgr)
        print(f"Saved {out_path}")

    # --- Grid 3: Random texture ---
    # Use solid dataset references for random textures
    print(f"\n--- Random texture samples ---")
    rand_ids = get_sample_ids(args.solid_dataset, count=args.rows, seed=args.seed + 2)
    rand_rows = []
    for i, sid in enumerate(rand_ids):
        try:
            ref, tex, mask, gt = load_sample(args.solid_dataset, sid)
            rand_tex = generate_random_texture(mask, seed=args.seed + 100 + i)
            gen = run_renderer(model, ref, rand_tex, mask, device)
            rand_rows.append({'images': [ref, rand_tex, mask, gt, gen], 'sample_id': sid})
            print(f"  Sample {sid}: OK")
        except FileNotFoundError as e:
            print(f"  Sample {sid}: SKIPPED ({e})")

    if rand_rows:
        rand_labels = ['Reference', 'Random Texture', 'Mask', 'Ground Truth', 'Generated']
        grid = make_grid(rand_rows, rand_labels)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, 'unet3_random.png')
        cv2.imwrite(out_path, grid_bgr)
        print(f"Saved {out_path}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
