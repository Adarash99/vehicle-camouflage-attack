# Vehicle Adversarial Camouflage Attack

Adversarial texture optimization that fools vehicle object detectors while producing natural-looking patterns. Uses the CARLA autonomous driving simulator, a U-Net neural renderer, and end-to-end differentiable adversarial optimization with Expectation over Transformation (EOT).

The attack optimizes a coarse 16x16 texture tile that, when repeated across the vehicle surface, minimises detection confidence across multiple viewpoints, distances, weather conditions, and scene backgrounds.

## Demo

https://github.com/user-attachments/assets/video.mp4

<video src="video.mp4" controls width="100%"></video>

## Results

| Metric | No Attack | DTA (CVPR 2022) | Ours |
|--------|-----------|-----------------|------|
| Max confidence | 0.99 | 0.34 | 0.116 |
| Mean confidence | 0.95 | — | 0.079 |

Evaluated on EfficientDet-D0 with transfer to YOLOv5, SSD, Faster R-CNN, and Mask R-CNN.

## Architecture

```
Texture (16x16) ──> Tile+Upsample (1024x1024) ──> U-Net Neural Renderer ──> EfficientDet-D0 ──> Attack Loss
  (requires_grad)      (nearest neighbour)            (8M params)              (pre-NMS)        -log(1 - conf)
       ^                                                                                            |
       └──────────────────────── loss.backward() <──────────────────────────────────────────────────┘
```

The neural renderer is trained to approximate CARLA's rendering engine, enabling fully differentiable texture-to-detection gradients without requiring a differentiable simulator.

## Prerequisites

- **OS:** Ubuntu 20.04+ (tested on 22.04)
- **GPU:** NVIDIA GPU with 24 GB VRAM (tested on RTX 3090), CUDA 11+
- **CARLA 0.9.16:** Download the [pre-built release](https://github.com/carla-simulator/carla/releases/tag/0.9.16/)
- **Conda:** Miniconda or Anaconda

## Setup

### 1. Install CARLA 0.9.16

Download and extract the CARLA 0.9.16 package:

```bash
# Extract to a directory of your choice
tar -xzf CARLA_0.9.16.tar.gz -C ~/CARLA_0.9.16

# Test that the server starts
~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen
```

### 2. Create the conda environment

```bash
conda create -n camo python=3.7 -y
conda activate camo

# PyTorch 1.13.1 with CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Core dependencies
pip install effdet==0.4.1 timm==0.9.12 opencv-python scipy Pillow

# CARLA Python API (from the CARLA installation)
pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp37-cp37m-linux_x86_64.whl
```

### 3. Clone and configure

```bash
git clone <this-repo-url>
cd vehicle-camouflage-attack

# Edit run.sh to set your CARLA path
# Change: export CARLA_PKG=/path/to/CARLA_0.9.16
```

### 4. Configure `run.sh`

All scripts are launched through `run.sh`, which handles process cleanup, conda activation, PYTHONPATH, and CARLA server startup. To run a different script, edit the Python command after the `# launch script` comment in `run.sh`.

## Reproducing the Results

### Step 1: Generate the solid-colour dataset (6400 train + 1600 val)

This dataset captures the vehicle under random solid-colour textures across varied viewpoints and lighting, providing the neural renderer with basic colour-to-rendering mappings.

```bash
# Edit run.sh launch command to:
cd scripts && python dataset_generation.py --output-dir ../dataset_8k/train --num-samples 6400 --resume

./run.sh

# Then for validation:
cd scripts && python dataset_generation.py --output-dir ../dataset_8k/val --num-samples 1600 --resume

./run.sh
```

Each sample produces four images (1024x1024):
- `reference/` — scene with neutral gray vehicle
- `texture/` — texture pattern on black background
- `rendered/` — scene with textured vehicle (ground truth)
- `mask/` — binary vehicle segmentation mask

Generation takes ~3-4 hours per split on a single GPU.

### Step 2: Generate the multi-colour composite dataset (6400 train + 1600 val)

This dataset extends training with multi-colour patterns (2-6 colours per sample), teaching the renderer to handle complex colour distributions closer to real adversarial textures.

```bash
# Edit run.sh launch command to:
cd scripts && python dataset_generation_multicolor.py --output-dir ../dataset_multicolor/train --num-samples 6400 --resume

./run.sh

# Then for validation:
cd scripts && python dataset_generation_multicolor.py --output-dir ../dataset_multicolor/val --num-samples 1600 --resume

./run.sh
```

### Step 3: Train the U-Net neural renderer

The renderer learns to map (reference image, texture, car mask) to a realistically rendered scene. It uses mask-weighted L1 loss combined with VGG perceptual loss.

```bash
# Edit run.sh launch command to:
python models/unet3/train_unet3.py \
    --datasets dataset_8k/train dataset_multicolor/train \
    --val-datasets dataset_8k/val dataset_multicolor/val \
    --epochs 100 --batch-size 4 --lambda-perceptual 0.1

./run.sh
```

Training takes ~8 hours on an RTX 3090. The best model is saved to `models/unet3/trained/best_model.pt`.

**Architecture:** 7-channel input (reference RGB + texture RGB + car mask), encoder-decoder with skip connections, mask-based output blending (`output = conv_output * mask + reference * (1 - mask)`), ~8M parameters.

### Step 4: Run adversarial texture optimisation

The attack optimises a 16x16 texture using EOT across:
- 12 yaw angles (every 30 degrees)
- 3 pitch angles (5, 10, 15 degrees)
- 4 distances (6, 8, 10, 12 m)
- 6 weather presets
- 50 spawn points (cycled every 10 iterations)

```bash
# Edit run.sh launch command to:
python experiments/phase1_random/train.py

./run.sh
```

Training runs for 1000 iterations (~2-3 hours). Outputs are saved to `experiments/phase1_random/`:
- `final/texture_final.npy` — optimised texture (16x16x3, float32, range [0, 1])
- `final/texture_final.png` — visual preview
- `training_log.csv` — per-iteration loss and confidence values
- `checkpoints/` — periodic texture snapshots
- `visualizations/` — rendered previews at each checkpoint

### Step 5: Evaluate the adversarial texture

#### Neural renderer evaluation (no CARLA source build needed)

```bash
# Edit run.sh launch command to:
python evaluation/TextureEvaluator.py \
    --mode neural \
    --texture experiments/phase1_random/final/texture_final.npy

./run.sh
```

#### CARLA direct rendering evaluation (requires source-built CARLA with static Tesla mesh)

For ground-truth evaluation, you need a source-built CARLA with a static Tesla Model 3 mesh placed in the UE4 editor. This bypasses the neural renderer and applies the texture directly in the simulator.

```bash
# Edit run.sh launch command to:
python evaluation/TextureEvaluator.py \
    --mode carla \
    --object-name SM_TeslaM3_v2 \
    --texture experiments/phase1_random/final/texture_final.npy

./run.sh
```

#### Transfer evaluation (test on other detectors)

Supported detectors: `efficientdet`, `yolov5s`, `yolov5m`, `yolov5l`, `ssd`, `faster_rcnn`, `mask_rcnn`

```bash
# Edit run.sh launch command to:
python evaluation/transfer_eval.py \
    --detector faster_rcnn \
    --object-name SM_TeslaM3_v2 \
    --texture experiments/phase1_random/final/texture_final.npy

./run.sh
```

## Project Structure

```
├── attack/                     # Attack pipeline (PyTorch)
│   ├── detector_pytorch.py     # EfficientDet-D0 with pre-NMS gradient access
│   ├── loss_pytorch.py         # Attack loss: -log(1 - max_confidence)
│   ├── eot_trainer_pytorch.py  # EOT trainer with autograd
│   └── texture_projection.py   # DTA-style repeated texture tiling
├── evaluation/                 # Evaluation scripts
│   ├── TextureEvaluator.py     # Single-mode evaluation (neural/carla)
│   ├── transfer_eval.py        # Cross-detector transfer evaluation
│   ├── compare_evaluator.py    # Neural vs CARLA side-by-side comparison
│   └── detectors/              # Detector wrappers (EfficientDet, YOLO, SSD, etc.)
├── experiments/
│   └── phase1_random/          # Phase 1 attack training
│       └── train.py
├── models/
│   ├── unet/
│   │   └── renderer_unet.py   # UNetRenderer nn.Module (~8M params)
│   └── unet3/
│       ├── train_unet3.py      # Training script (L1 + VGG loss)
│       ├── renderer_dataset.py # PyTorch Dataset
│       └── trained/            # Model weights (not in repo)
├── scripts/
│   ├── CarlaHandler.py         # CARLA simulator interface
│   ├── dataset_generation.py   # Solid-colour dataset generation
│   ├── dataset_generation_multicolor.py  # Multi-colour dataset generation
│   └── texture_applicator_pytorch.py     # Differentiable texture application
├── test_scripts/               # Unit tests for pipeline components
├── textures/                   # Reference textures and baselines
└── run.sh                      # Main launcher script
```

## Citation

If you use this code, please cite our paper (details to be added after publication).
