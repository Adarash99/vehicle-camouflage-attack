import numpy as np
import cv2
import time
import os
import random
import argparse
import json
from CarlaHandler import *


def parse_args():
    parser = argparse.ArgumentParser(description='Generate renderer training dataset')
    parser.add_argument('--output-dir', default='./dataset', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=6400, help='Number of samples')
    parser.add_argument('--start-index', type=int, default=0, help='Starting file index')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
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

# Configurable Parameters
vehicle_id = 'vehicle.tesla.model3'
res = 1024  # Updated from 500 to 1024 for higher resolution
ref_color = (124, 124, 124)  # BGR format
patch_size = 1024  # Size of the adversarial patch (updated from 500)


def main():
    args = parse_args()

    # Initialize counters
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
            print(f"📂 Resuming from file_n={file_n}, sample_n={sample_n}/{total_sample_n}")
        else:
            print(f"📂 No progress file found, starting fresh")

    # Initialize CARLA
    handler, n_spawn_points = initialize_carla()

    

    while True:
        # Check if the number of samples is reached
        if sample_n == total_sample_n:
            print("Sample limit reached. Exiting...")
            break

        try:
            # Randomize parameters
            handler.change_spawn_point(random.randint(1, n_spawn_points-1))
            handler.update_distance(random.randint(5, 10))
            handler.update_pitch(random.randint(0, 60))
            handler.update_yaw(random.randint(0, 359))
            handler.update_sun_altitude_angle(random.randint(20, 150))
            handler.update_sun_azimuth_angle(random.randint(0, 360))
            handler.world_tick(100)
            time.sleep(0.1)

            # Generate random color
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # 1. Get reference image (original color)
            handler.change_vehicle_color(ref_color)
            handler.world_tick(100)
            time.sleep(0.1)
            
            # Get images
            image = handler.get_image()  # RGB format
            seg_image = handler.get_segmentation()  # BGR format
            
            
            # Create vehicle mask (blue pixels in segmentation)
            vehicle_mask = (seg_image[:,:,0] == 255) & \
                        (seg_image[:,:,1] == 0) & \
                        (seg_image[:,:,2] == 0)
            
            # Validate vehicle mask
            if not validate_car_mask(vehicle_mask):
                print(f"⚠️ Sample {sample_n}: Invalid mask (multiple cars or no car detected)")
                continue

            # Save reference image with full background visible
            reference_image = image.copy()
            cv2.imwrite(os.path.join(output_dir, 'reference', f'{file_n}.png'), reference_image)

            # Use full vehicle mask from semantic segmentation (pixel-perfect)
            # vehicle_mask is already computed above from segmentation
            mask_uint8 = vehicle_mask.astype(np.uint8) * 255

            # Validate mask (must have reasonable coverage)
            mask_pixels = vehicle_mask.sum()
            if mask_pixels < 1000:
                print(f"⚠️ Sample {sample_n}: Vehicle mask too small ({mask_pixels} pixels)")
                continue

            # Save mask as grayscale image
            cv2.imwrite(os.path.join(output_dir, 'mask', f'{file_n}.png'), mask_uint8)

            # 2. Create texture image (black background with rand_color car)
            texture_image = np.zeros_like(seg_image)  # Black background
            texture_image[vehicle_mask] = rand_color  # Apply random color to car pixels
            cv2.imwrite(os.path.join(output_dir, 'texture', f'{file_n}.png'), cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))

            # 3. Get rendered image (actual RGB appearance with new color)
            handler.change_vehicle_color(rand_color)
            handler.world_tick(100)
            time.sleep(0.1)  # Allow time for color change to render
            
            # Get the actual rendered RGB image
            rendered_rgb = handler.get_image()  # RGB format
            
            # Save rendered image with full background visible
            rendered_image = rendered_rgb.copy()
            cv2.imwrite(os.path.join(output_dir, 'rendered', f'{file_n}.png'), rendered_image)

            print(f"✅ Sample {sample_n}/{total_sample_n}: Generated reference, texture, rendered, and mask images.")

            sample_n += 1
            file_n += 1

            # Save progress periodically (every 10 samples)
            if sample_n % 10 == 0:
                save_progress(output_dir, file_n, sample_n)

        except Exception as e:
            print(f"Error generating sample {sample_n}: {e}")
            continue

    # Save final progress
    save_progress(output_dir, file_n, sample_n)
    print(f"✅ Generation complete: {sample_n} samples saved to {output_dir}")

    if 'handler' in locals():
        handler.destroy_all_vehicles()
        del handler
        print("Cleanup completed")


def initialize_carla():
        """
        Initialize CARLA and spawn a vehicle.
        """

        handler = CarlaHandler(x_res=res, y_res=res, town='Town03')
        handler.world_tick(10)
        handler.destroy_all_vehicles()
        handler.world_tick(100)

        # Spawn vehicle
        handler.spawn_vehicle(vehicle_id)
        handler.update_view('3d')
        n_spawn_points = handler.get_spawn_points()
        return handler, n_spawn_points


def validate_car_mask(mask):
    """
    Returns True if mask contains exactly one connected car region.
    - mask: Binary vehicle mask (True/False or 0/255)
    """
    # Convert to uint8 if needed
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # Background counts as label 0, so:
    # num_labels = 1 (only background) → No car
    # num_labels = 2 → Exactly one car
    # num_labels > 2 → Multiple cars/objects
    return num_labels == 2  # Exactly one car


if __name__ == '__main__':
    main()