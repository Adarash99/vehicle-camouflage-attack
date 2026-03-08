"""
Adversarial Attack Pipeline (Pure PyTorch)

This package implements the adversarial camouflage attack system for
autonomous vehicle object detection evasion.

Components:
- detector_pytorch: PyTorch EfficientDet with pre-NMS access and gradient support
- loss_pytorch: PyTorch attack loss functions
- eot_trainer_pytorch: EOT trainer with true autograd (no finite differences)
- logger: CSV logging utilities

Author: Adversarial Camouflage Research Project
Date: 2026-02-04
"""

from .detector_pytorch import EfficientDetPyTorch
from .loss_pytorch import attack_loss_pytorch, attack_loss_with_stats_pytorch
from .eot_trainer_pytorch import EOTTrainerPyTorch, create_viewpoint_configs
from .logger import CSVLogger

__all__ = [
    # Detector
    'EfficientDetPyTorch',
    # Loss functions
    'attack_loss_pytorch',
    'attack_loss_with_stats_pytorch',
    # EOT Trainer
    'EOTTrainerPyTorch',
    'create_viewpoint_configs',
    # Utilities
    'CSVLogger',
]
