import numpy as np
import cv2
import time
import os
import random
import argparse
import json
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from CarlaHandler import *


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multi-color composite renderer training dataset')
    parser.add_argument('--output-dir', default='./dataset_multicolor', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=4000, help='Number of samples')
    parser.add_argument('--start-index', type=int, default=0, help='Starting file index')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    parser.add_argument('--min-colors', type=int, default=2, help='Minimum number of colors per sample')
    parser.add_argument('--max-colors', type=int, default=6, help='Maximum number of colors per sample')
    return parser.parse_args()


def save_progress(output_dir, file_n, sample_n):
    """Save generation progress for resumability."""
    progress_file = os.path.join(output_dir, '.progress.json')
    with open(progress_file, 'w') as f:
        json.dump({'file_n': file_n, 'sample_n': sample_n}, f)


def load_progress(output_dir):
    """Load generation progress from file."""
    progress_file = os.path.join(output_dir, '.progress.json')
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return None


# --- Pattern mask generators ---
# Each returns a binary (H, W) bool array

def make_checkerboard_mask(h, w):
    """Checkerboard pattern with random grid size and rotation."""
    # grid_size = number of cells across (e.g. 4 means 4x4 cells)
    grid_size = random.choice([2, 4, 8, 16, 32, 64])
    cell_h = h // grid_size
    cell_w = w // grid_size

    mask = np.zeros((h, w), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            if (r + c) % 2 == 1:
                y0 = r * cell_h
                y1 = (r + 1) * cell_h if r < grid_size - 1 else h
                x0 = c * cell_w
                x1 = (c + 1) * cell_w if c < grid_size - 1 else w
                mask[y0:y1, x0:x1] = 255

    # Random rotation
    angle = random.uniform(0, 360)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return mask > 127


def make_stripes_mask(h, w):
    """Horizontal stripes with random width and rotation."""
    stripe_width = random.randint(16, 256)
    mask = np.zeros((h, w), dtype=np.uint8)
    y = 0
    fill = False
    while y < h:
        if fill:
            mask[y:min(y + stripe_width, h), :] = 255
        fill = not fill
        y += stripe_width

    # Random rotation
    angle = random.uniform(0, 360)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return mask > 127


def make_blobs_mask(h, w):
    """Random blobs via gaussian-filtered noise."""
    sigma = random.choice([8, 16, 32, 64])
    noise = np.random.rand(h, w).astype(np.float32)
    smoothed = gaussian_filter(noise, sigma=sigma)
    return smoothed > 0.5


def make_halfplane_mask(h, w, car_mask=None):
    """Half-plane through car centroid at random angle."""
    if car_mask is not None and car_mask.any():
        ys, xs = np.where(car_mask)
        cx, cy = xs.mean(), ys.mean()
    else:
        cx, cy = w / 2, h / 2

    angle = random.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[:h, :w]
    return (np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy)) > 0


def make_voronoi_mask(h, w, car_mask=None):
    """Voronoi regions with random seed points, alternating even/odd."""
    n_seeds = random.randint(5, 20)

    if car_mask is not None and car_mask.any():
        # Place seeds within the car mask
        ys, xs = np.where(car_mask)
        indices = np.random.choice(len(xs), size=min(n_seeds, len(xs)), replace=False)
        seeds = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float64)
    else:
        seeds = np.column_stack([
            np.random.randint(0, w, n_seeds),
            np.random.randint(0, h, n_seeds)
        ]).astype(np.float64)

    tree = KDTree(seeds)

    # Query all pixel coordinates
    yy, xx = np.mgrid[:h, :w]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    _, labels = tree.query(coords)
    labels = labels.reshape(h, w)

    return (labels % 2) == 1


MASK_GENERATORS = [
    'checkerboard',
    'stripes',
    'blobs',
    'halfplane',
    'voronoi',
]


def generate_pattern_mask(h, w, car_mask=None):
    """Generate a random pattern mask of a random type."""
    mask_type = random.choice(MASK_GENERATORS)

    if mask_type == 'checkerboard':
        return make_checkerboard_mask(h, w)
    elif mask_type == 'stripes':
        return make_stripes_mask(h, w)
    elif mask_type == 'blobs':
        return make_blobs_mask(h, w)
    elif mask_type == 'halfplane':
        return make_halfplane_mask(h, w, car_mask)
    elif mask_type == 'voronoi':
        return make_voronoi_mask(h, w, car_mask)


# Configurable Parameters
vehicle_id = 'vehicle.tesla.model3'
res = 1024
ref_color = (124, 124, 124)  # BGR format


def main():
    args = parse_args()

    file_n = args.start_index
    total_sample_n = args.num_samples
    sample_n = 0
    output_dir = args.output_dir

    # Create directories
    os.makedirs(os.path.join(output_dir, 'reference'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'texture'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rendered'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)

    # Resume from progress if requested
    if args.resume:
        progress = load_progress(output_dir)
        if progress:
            file_n = progress['file_n']
            sample_n = progress['sample_n']
            print(f"Resuming from file_n={file_n}, sample_n={sample_n}/{total_sample_n}")
        else:
            print(f"No progress file found, starting fresh")

    # Initialize CARLA
    handler, n_spawn_points = initialize_carla()

    while True:
        if sample_n == total_sample_n:
            print("Sample limit reached. Exiting...")
            break

        try:
            # Randomize scene parameters
            handler.change_spawn_point(random.randint(1, n_spawn_points - 1))
            handler.update_distance(random.randint(5, 10))
            handler.update_pitch(random.randint(0, 60))
            handler.update_yaw(random.randint(0, 359))
            handler.update_sun_altitude_angle(random.randint(20, 150))
            handler.update_sun_azimuth_angle(random.randint(0, 360))
            handler.world_tick(100)
            time.sleep(0.1)

            # 1. Capture reference image (grey car)
            handler.change_vehicle_color(ref_color)
            handler.world_tick(100)
            time.sleep(0.1)

            image = handler.get_image()  # RGB format
            seg_image = handler.get_segmentation()  # BGR format

            # Create vehicle mask (blue pixels in segmentation)
            vehicle_mask = (seg_image[:, :, 0] == 255) & \
                           (seg_image[:, :, 1] == 0) & \
                           (seg_image[:, :, 2] == 0)

            # Validate vehicle mask
            if not validate_car_mask(vehicle_mask):
                print(f"Sample {sample_n}: Invalid mask (multiple cars or no car detected)")
                continue

            mask_pixels = vehicle_mask.sum()
            if mask_pixels < 1000:
                print(f"Sample {sample_n}: Vehicle mask too small ({mask_pixels} pixels)")
                continue

            reference_image = image.copy()

            # 2. Determine number of colors for this sample
            n_colors = random.randint(args.min_colors, args.max_colors)

            # 3. Generate N random colors and capture CARLA renders for each
            colors = []
            renders = []
            for _ in range(n_colors):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                colors.append(color)

                handler.change_vehicle_color(color)
                handler.world_tick(100)
                time.sleep(0.1)
                render = handler.get_image()  # RGB format
                renders.append(render)

            # 4. Generate N-1 pattern masks for layered compositing
            h, w = reference_image.shape[:2]
            pattern_masks = []
            for _ in range(n_colors - 1):
                pmask = generate_pattern_mask(h, w, car_mask=vehicle_mask)
                pattern_masks.append(pmask)

            # 5. Layered compositing
            # Start with first color as base
            rendered_composite = renders[0].copy()
            texture_composite = np.zeros_like(reference_image)
            texture_composite[vehicle_mask] = colors[0]

            # Layer subsequent colors on top using pattern masks
            for i in range(1, n_colors):
                layer_mask = pattern_masks[i - 1] & vehicle_mask
                rendered_composite[layer_mask] = renders[i][layer_mask]
                texture_composite[layer_mask] = colors[i]

            # 6. Save outputs
            mask_uint8 = vehicle_mask.astype(np.uint8) * 255

            cv2.imwrite(os.path.join(output_dir, 'reference', f'{file_n}.png'), reference_image)
            cv2.imwrite(os.path.join(output_dir, 'texture', f'{file_n}.png'),
                        cv2.cvtColor(texture_composite, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, 'rendered', f'{file_n}.png'), rendered_composite)
            cv2.imwrite(os.path.join(output_dir, 'mask', f'{file_n}.png'), mask_uint8)

            print(f"Sample {sample_n}/{total_sample_n}: Generated {n_colors}-color composite")

            sample_n += 1
            file_n += 1

            # Save progress periodically
            if sample_n % 10 == 0:
                save_progress(output_dir, file_n, sample_n)

        except Exception as e:
            print(f"Error generating sample {sample_n}: {e}")
            continue

    # Save final progress
    save_progress(output_dir, file_n, sample_n)
    print(f"Generation complete: {sample_n} samples saved to {output_dir}")

    if 'handler' in locals():
        handler.destroy_all_vehicles()
        del handler
        print("Cleanup completed")


def initialize_carla():
    """Initialize CARLA and spawn a vehicle."""
    handler = CarlaHandler(x_res=res, y_res=res, town='Town03')
    handler.world_tick(10)
    handler.destroy_all_vehicles()
    handler.world_tick(100)

    handler.spawn_vehicle(vehicle_id)
    handler.update_view('3d')
    n_spawn_points = handler.get_spawn_points()
    return handler, n_spawn_points


def validate_car_mask(mask):
    """Returns True if mask contains exactly one connected car region."""
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    return num_labels == 2


if __name__ == '__main__':
    main()
