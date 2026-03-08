#!/usr/bin/env python3
"""Export adversarial texture as a PNG for UE4 import."""
import sys, os
import numpy as np
from PIL import Image

texture_path = sys.argv[1] if len(sys.argv) > 1 else 'experiments/phase1_eot_pytorch/final/texture_final.npy'
output_path = sys.argv[2] if len(sys.argv) > 2 else 'experiments/phase1_eot_pytorch/final/texture_final.png'

# Load coarse texture [3, 16, 16]
tex = np.load(texture_path)
print(f"Loaded: {tex.shape}, range [{tex.min():.3f}, {tex.max():.3f}]")

# Tile to 1024x1024 (16x16 tiles of 16x16 = 256x256, then tile 4x to 1024)
# Actually: tile_count = coarse_size = 16, so 16x16 repeated 16 times = 256x256,
# then nearest upsample to 1024x1024
tile_count = tex.shape[1]  # 16
tiled = np.tile(tex, (1, tile_count, tile_count))  # [3, 256, 256]

# Upsample to 1024x1024 via nearest neighbor
from PIL import Image
img_hwc = np.transpose(tiled, (1, 2, 0))  # [256, 256, 3]
img_uint8 = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)
img = Image.fromarray(img_uint8)
img_1024 = img.resize((1024, 1024), Image.NEAREST)
img_1024.save(output_path)
print(f"Saved tiled 1024x1024 to: {output_path}")

# Also save at native tiled size (256x256) in case that's useful
output_256 = output_path.replace('.png', '_256.png')
img.save(output_256)
print(f"Saved tiled 256x256 to: {output_256}")

# Also save the raw 16x16 coarse texture
output_coarse = output_path.replace('.png', '_16x16.png')
coarse_hwc = np.transpose(tex, (1, 2, 0))
coarse_uint8 = (np.clip(coarse_hwc, 0, 1) * 255).astype(np.uint8)
Image.fromarray(coarse_uint8).save(output_coarse)
print(f"Saved coarse 16x16 to: {output_coarse}")
