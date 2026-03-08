#!/usr/bin/env python
"""Verify dataset integrity: check file counts and image validity."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def verify_dataset(dataset_dir: str, expected_count: int) -> bool:
    """
    Verify dataset integrity.

    Args:
        dataset_dir: Path to dataset directory (e.g., dataset_8k/train)
        expected_count: Expected number of images per subdirectory

    Returns:
        True if dataset is valid, False otherwise
    """
    dataset_path = Path(dataset_dir)
    subdirs = ['reference', 'texture', 'rendered', 'mask']

    if not dataset_path.exists():
        print(f"❌ Dataset directory does not exist: {dataset_dir}")
        return False

    all_valid = True
    file_counts = {}

    # Check each subdirectory
    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if not subdir_path.exists():
            print(f"❌ Missing subdirectory: {subdir}")
            all_valid = False
            continue

        # Get all PNG files
        files = sorted(subdir_path.glob('*.png'))
        file_counts[subdir] = len(files)

        if len(files) != expected_count:
            print(f"⚠️ {subdir}: Found {len(files)} files, expected {expected_count}")
            all_valid = False
        else:
            print(f"✅ {subdir}: {len(files)} files")

    # Check that all subdirs have matching filenames
    if all(subdir in file_counts for subdir in subdirs):
        reference_files = set(f.name for f in (dataset_path / 'reference').glob('*.png'))
        for subdir in subdirs[1:]:
            subdir_files = set(f.name for f in (dataset_path / subdir).glob('*.png'))
            missing = reference_files - subdir_files
            extra = subdir_files - reference_files
            if missing:
                print(f"⚠️ {subdir}: Missing files: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                all_valid = False
            if extra:
                print(f"⚠️ {subdir}: Extra files: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
                all_valid = False

    # Sample check: verify a few random images can be loaded
    print("\nSampling image validity...")
    corrupt_files = []
    reference_path = dataset_path / 'reference'
    if reference_path.exists():
        files = list(reference_path.glob('*.png'))
        sample_size = min(100, len(files))
        sample_indices = np.random.choice(len(files), sample_size, replace=False)

        for idx in sample_indices:
            filepath = files[idx]
            try:
                img = cv2.imread(str(filepath))
                if img is None:
                    corrupt_files.append(filepath.name)
            except Exception as e:
                corrupt_files.append(f"{filepath.name}: {e}")

        if corrupt_files:
            print(f"❌ Found {len(corrupt_files)} corrupt images: {corrupt_files[:5]}")
            all_valid = False
        else:
            print(f"✅ Sampled {sample_size} images, all valid")

    return all_valid


def main():
    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('expected_count', type=int, help='Expected number of images')
    args = parser.parse_args()

    print(f"Verifying dataset: {args.dataset_dir}")
    print(f"Expected count: {args.expected_count}")
    print("-" * 40)

    is_valid = verify_dataset(args.dataset_dir, args.expected_count)

    print("-" * 40)
    if is_valid:
        print("✅ Dataset verification PASSED")
        sys.exit(0)
    else:
        print("❌ Dataset verification FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
