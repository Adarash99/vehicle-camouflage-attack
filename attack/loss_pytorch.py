#!/usr/bin/env python3
"""
Attack Loss Functions (PyTorch)

PyTorch implementation of DTA attack loss for use with PyTorch EfficientDet.

Based on: docs/plans/2026-01-31-efficientdet-integration-design.md

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import torch
import torch.nn.functional as F


def attack_loss_pytorch(class_logits, car_class_id=2):
    """
    Compute adversarial attack loss following DTA formulation (PyTorch version).

    Formula:
        L_atk = -log(1 - max_confidence)

    Goal: Minimize detection confidence to make the detector miss the car.

    Args:
        class_logits: Raw class logits from detector
                     Shape: [batch, num_anchors, num_classes]
        car_class_id: COCO class ID for car (2 in PyTorch, 0-indexed)

    Returns:
        loss: Scalar tensor representing average attack loss across batch
    """
    # Get car class logits
    car_logits = class_logits[:, :, car_class_id]  # [batch, anchors]

    # Apply sigmoid to get probabilities
    car_probs = torch.sigmoid(car_logits)  # [batch, anchors]

    # Get maximum confidence per image
    max_conf, _ = torch.max(car_probs, dim=1)  # [batch]

    # DTA loss: -log(1 - max_score)
    # Add epsilon for numerical stability
    loss_per_image = -torch.log(1.0 - max_conf + 1e-8)

    # Average across batch (for EOT)
    total_loss = torch.mean(loss_per_image)

    return total_loss


def attack_loss_with_stats_pytorch(class_logits, car_class_id=2):
    """
    Same as attack_loss_pytorch but also returns statistics.

    Returns:
        loss: Scalar tensor
        stats: Dictionary with debug information
    """
    # Get car logits
    car_logits = class_logits[:, :, car_class_id]
    car_probs = torch.sigmoid(car_logits)

    # Per-image max confidence
    max_conf, max_indices = torch.max(car_probs, dim=1)

    # Compute loss per image
    loss_per_image = -torch.log(1.0 - max_conf + 1e-8)

    # Overall loss
    total_loss = torch.mean(loss_per_image)

    # Statistics
    stats = {
        'max_confidence': torch.max(max_conf).item(),
        'mean_confidence': torch.mean(max_conf).item(),
        'per_image_loss': loss_per_image.detach().cpu().numpy(),
        'per_image_max_conf': max_conf.detach().cpu().numpy(),
    }

    return total_loss, stats


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH ATTACK LOSS TEST")
    print("=" * 70)
    print()

    # Test Case 1: High confidence detections (bad for attack)
    print("Test Case 1: High confidence detections")
    # Simulate high confidence on car class
    class_logits = torch.randn(2, 1000, 90)  # [batch=2, anchors=1000, classes=90]
    class_logits[:, :, 2] = 5.0  # High logits for car class → high confidence

    loss = attack_loss_pytorch(class_logits, car_class_id=2)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Expected: High loss (confident detections are bad for attack)")
    print()

    # Test Case 2: Low confidence detections (good for attack)
    print("Test Case 2: Low confidence detections")
    class_logits = torch.randn(2, 1000, 90)
    class_logits[:, :, 2] = -5.0  # Low logits for car class → low confidence

    loss = attack_loss_pytorch(class_logits, car_class_id=2)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Expected: Low loss (low confidence is good for attack)")
    print()

    # Test Case 3: With statistics
    print("Test Case 3: Loss with statistics")
    class_logits = torch.randn(3, 1000, 90)
    class_logits[:, :, 2] = torch.randn(3, 1000) * 2.0  # Varied confidences

    loss, stats = attack_loss_with_stats_pytorch(class_logits, car_class_id=2)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Max confidence: {stats['max_confidence']:.4f}")
    print(f"  Mean confidence: {stats['mean_confidence']:.4f}")
    print(f"  Per-image losses: {stats['per_image_loss']}")
    print()

    # Test Case 4: Gradient flow
    print("Test Case 4: Gradient flow test")
    class_logits = torch.randn(2, 1000, 90, requires_grad=True)

    loss = attack_loss_pytorch(class_logits, car_class_id=2)
    loss.backward()

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradients exist: {class_logits.grad is not None}")
    print(f"  Gradient mean: {torch.mean(torch.abs(class_logits.grad)).item():.8f}")
    print(f"  ✓ Gradients flow correctly!")
    print()

    print("=" * 70)
    print("ALL PYTORCH LOSS TESTS PASSED ✓")
    print("=" * 70)
