#!/bin/bash

##################################
pkill -9 Carla
pkill -9 python
##################################

sleep 1

export ROOT=/home/adarash/camouflage
export PORT=2000
export PYTHONPATH="${ROOT}":$PYTHONPATH

# Packaged CARLA 0.9.16 (stock vehicle materials, for training)
export CARLA_PKG=/home/adarash/CARLA_0.9.16
# Source-built CARLA (modified Tesla mesh, for evaluation only)
# export CARLA_SRC=<path-to-source-build>

# Start packaged CARLA server
#${CARLA_PKG}/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=${PORT} &
#CARLA_PID=$!
#echo "Started packaged CARLA (PID: $CARLA_PID), waiting for server..."
#sleep 15

# launch script
#python evaluation/TextureEvaluator.py --mode carla --object-name SM_TeslaM3_v2 --texture experiments/phase3_town01/final/texture_final.npy --cloudiness 60 --fps 20 --output-dir evaluation/results/phase3/efficientdet

#TEXTURE=experiments/phase3_town01/final/texture_final.npy
#COMMON_ARGS="--object-name SM_TeslaM3_v2 --texture $TEXTURE --cloudiness 60 --fps 20"

#python evaluation/transfer_eval.py --detector yolov5s $COMMON_ARGS --output-dir evaluation/results/phase3/transfer-yolov5s

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5m $COMMON_ARGS --output-dir evaluation/results/phase3/transfer-yolov5m

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5l $COMMON_ARGS --output-dir evaluation/results/phase3/transfer-yolov5l

#sleep 10

#python evaluation/transfer_eval.py --detector ssd $COMMON_ARGS --output-dir evaluation/results/phase3/transfer-ssd

#sleep 10

#python evaluation/transfer_eval.py --detector faster_rcnn $COMMON_ARGS --output-dir evaluation/results/phase3/transfer-faster-rcnn

#sleep 10

#python evaluation/transfer_eval.py --detector mask_rcnn $COMMON_ARGS --output-dir evaluation/results/phase3/transfer-mask-rcnn

#python evaluation/TextureEvaluator.py --mode carla --object-name SM_TeslaM3_v2 --texture textures/dta.npy --cloudiness 60 --fps 20 --output-dir evaluation/results/baselines/dta-efficientdet

#TEXTURE=textures/dta.npy
#COMMON_ARGS="--object-name SM_TeslaM3_v2 --texture $TEXTURE --cloudiness 60 --fps 20"

#python evaluation/transfer_eval.py --detector yolov5s $COMMON_ARGS --output-dir evaluation/results/baselines/dta-yolov5s

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5m $COMMON_ARGS --output-dir evaluation/results/baselines/dta-yolov5m

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5l $COMMON_ARGS --output-dir evaluation/results/baselines/dta-yolov5l

#sleep 10

#python evaluation/transfer_eval.py --detector ssd $COMMON_ARGS --output-dir evaluation/results/baselines/dta-ssd

#sleep 10

#python evaluation/transfer_eval.py --detector faster_rcnn $COMMON_ARGS --output-dir evaluation/results/baselines/dta-faster-rcnn

#sleep 10

#python evaluation/transfer_eval.py --detector mask_rcnn $COMMON_ARGS --output-dir evaluation/results/baselines/dta-mask-rcnn

#TEXTURE=experiments/phase1_eot_pytorch/final/random_texture.npy
#COMMON_ARGS="--object-name SM_TeslaM3_v2 --texture $TEXTURE --cloudiness 60 --fps 20"

#python evaluation/transfer_eval.py --detector yolov5m $COMMON_ARGS --output-dir evaluation/results/baselines/baseline-random-yolov5m

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5l $COMMON_ARGS --output-dir evaluation/results/baselines/baseline-random-yolov5l

#TEXTURE=experiments/phase1_eot_pytorch/final/random_texture.npy
#COMMON_ARGS="--object-name SM_TeslaM3_v2 --texture $TEXTURE --cloudiness 60 --fps 20"

#python evaluation/transfer_eval.py --detector yolov5m $COMMON_ARGS --output-dir evaluation/results/baselines/baseline-yolov5m

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5l $COMMON_ARGS --output-dir evaluation/results/baselines/baseline-yolov5l

# Compare neural renderer vs CARLA at 1-degree resolution
#python evaluation/compare_evaluator.py --object-name SM_TeslaM3_v2 --texture experiments/phase2_robust_eot/final/texture_final.npy --cloudiness 60 --fps 20 --yaw-step 1 --output-dir evaluation/results/phase2/compare-1-degree

# Re-evaluate phase2 texture with cloudiness=60 to match training conditions
#python evaluation/TextureEvaluator.py --mode carla --object-name SM_TeslaM3_v2 --texture experiments/phase2_robust_eot/final/texture_final.npy --cloudiness 60 --fps 20 --output-dir evaluation/results/phase2/efficientdet1

# Phase 2 re-evaluation with consistent cloudiness=60
TEXTURE=experiments/phase2_robust_eot/final/texture_final.npy
COMMON_ARGS="--object-name SM_TeslaM3_v2 --texture $TEXTURE --cloudiness 60 --fps 20"

python evaluation/TextureEvaluator.py --mode carla $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/efficientdet

sleep 10

python evaluation/transfer_eval.py --detector yolov5s $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/transfer-yolov5s

sleep 10

python evaluation/transfer_eval.py --detector yolov5m $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/transfer-yolov5m

sleep 10

python evaluation/transfer_eval.py --detector yolov5l $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/transfer-yolov5l

sleep 10

python evaluation/transfer_eval.py --detector ssd $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/transfer-ssd

sleep 10

python evaluation/transfer_eval.py --detector faster_rcnn $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/transfer-faster-rcnn

sleep 10

python evaluation/transfer_eval.py --detector mask_rcnn $COMMON_ARGS --output-dir evaluation/results/phase2_cloudiness60/transfer-mask-rcnn

#cd $ROOT && conda run -n camo1 python evaluation/compare_evaluator.py --object-name SM_TeslaM3_v2 --texture experiments/phase2_robust_eot/final/texture_final.npy --cloudiness 50 --fps 20 --output-dir evaluation/results/phase2/compare

# Phase 3: Robust EOT on Town01
#python experiments/phase3_town01/train.py --debug

#TEXTURE=experiments/phase2_robust_eot/final/texture_final.npy
#COMMON_ARGS="--object-name SM_TeslaM3_v2 --texture $TEXTURE --cloudiness 50 --fps 20"

#python evaluation/transfer_eval.py --detector efficientdet $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-efficientdet

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5s $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-yolov5s

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5m $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-yolov5m

#sleep 10

#python evaluation/transfer_eval.py --detector yolov5l $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-yolov5l

#sleep 10

#python evaluation/transfer_eval.py --detector ssd $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-ssd

#sleep 10

#python evaluation/transfer_eval.py --detector faster_rcnn $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-faster-rcnn

#sleep 10

#python evaluation/transfer_eval.py --detector mask_rcnn $COMMON_ARGS --output-dir evaluation/results/phase2/transfer-mask-rcnn
#python experiments/phase2_robust_eot/train.py --debug

# Normal car baselines (no texture modification, normal paint applied in CARLA)
#cd $ROOT && conda run -n camo1 python evaluation/TextureEvaluator.py --mode carla --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --skip-load-world --cloudiness 50 --fps 20 --output-dir evaluation/results/baseline-efficientdet

#python experiments/phase2_robust_eot/train.py --debug --skip-load-world

#sleep 10

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector ssd --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 50 --output-dir evaluation/results/baseline-ssd

#sleep 10

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector faster_rcnn --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 50 --output-dir evaluation/results/baseline-faster-rcnn

#sleep 10

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector mask_rcnn --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 50 --output-dir evaluation/results/baseline-mask-rcnn

#python evaluation/transfer_eval.py --detector yolov5s --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/random_texture.npy --cloudiness 50 --output-dir evaluation/results/baseline-yolo

#sleep 10
#
#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector ssd --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/random_texture.npy --cloudiness 50 --output-dir evaluation/results/baseline-random-ssd

#sleep 10
#
#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector faster_rcnn --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/random_texture.npy --cloudiness 50 --output-dir evaluation/results/baseline-random-faster-rcnn

#sleep 10

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector mask_rcnn --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/random_texture.npy --cloudiness 50 --output-dir evaluation/results/baseline-random-mask-rcnn

#cd $ROOT && conda run -n camo1 python evaluation/TextureEvaluator.py --mode carla --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/random_texture.npy --skip-load-world --cloudiness 50 --fps 20 --output-dir evaluation/results/baseline-random

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector mask_rcnn --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 50 --output-dir evaluation/results/transfer-mask-rcnn

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector faster_rcnn --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 50 --output-dir evaluation/results/transfer-faster-rcnn

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector ssd --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 50 --output-dir evaluation/results/transfer-ssd

#cd $ROOT && conda run -n camo1 python evaluation/transfer_eval.py --detector yolov5s --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --cloudiness 10

#N=1; while [ -d "$ROOT/evaluation/results/carla$N" ]; do N=$((N+1)); done; OUTDIR="evaluation/results/carla$N"
#cd $ROOT && conda run -n camo1 python evaluation/TextureEvaluator.py --mode carla --object-name SM_TeslaM3_v2 --texture experiments/phase1_eot_pytorch/final/texture_final.npy --skip-load-world --cloudiness 10 --fps 20 --output-dir "$OUTDIR"

#cd $ROOT && conda run -n camo1 python scripts/TextureEvaluator.py --mode neural --texture experiments/phase1_eot_pytorch/final/texture_final.npy
#cd $ROOT && conda run -n camo1 python scripts/probe_texture_api.py
#cd $ROOT && conda run -n camo1 python experiments/phase1_random_pytorch.py
#cd $ROOT && conda run -n camo1 python scripts/RendererVisualizer.py
#cd $ROOT && conda run -n camo1 python scripts/TriplanarVisualizer.py

# Previous commands (completed/archived)
# cd $ROOT && python models/unet3/train_unet3.py \
#     --datasets dataset_8k_revised/train dataset_multicolor/train \
#     --val-datasets dataset_8k_revised/val dataset_multicolor/val \
#     --epochs 100 --batch-size 10 --lambda-perceptual 0.1

# Previous commands (completed/archived)
# cd $ROOT/scripts && python dataset_generation_multicolor.py --output-dir $ROOT/dataset_multicolor/train --num-samples 6400 --resume
# cd $ROOT/scripts && python dataset_generation_multicolor.py --output-dir $ROOT/dataset_multicolor/val --num-samples 1600

# Previous commands (completed/archived)
# python models/unet2/train_unet2.py --dataset dataset_8k_revised/train --val-dataset dataset_8k_revised/val --epochs 100 --batch-size 10

# Previous commands (completed/archived)
# python models/unet/train_unet.py --dataset dataset_8k_revised/train --val-dataset dataset_8k_revised/val --epochs 100 --batch-size 8

# Previous commands (completed/archived)
# python car_segmentation.py --output-dir dataset_8k_revised/train --num-samples 6400 --resume
# python car_segmentation.py --output-dir dataset_8k_revised/val --num-samples 1600 --resume
# python car_segmentation.py --output-dir dataset_8k/train --num-samples 6400 --resume
# python car_segmentation.py --output-dir dataset_8k/val --num-samples 1600 --resume
# python models/train_renderer.py --dataset dataset_8k/train --val-dataset dataset_8k/val --epochs 100 --batch-size 10
