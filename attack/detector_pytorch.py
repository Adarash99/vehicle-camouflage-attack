#!/usr/bin/env python3
"""
PyTorch EfficientDet Wrapper with Pre-NMS Access

Uses Ross Wightman's effdet library to access raw detection outputs
BEFORE Non-Max Suppression, enabling proper gradient flow for
adversarial attacks.

Based on: docs/plans/2026-01-31-nms-gradient-issue.md

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from effdet import create_model, get_efficientdet_config
from effdet.anchors import Anchors, AnchorLabeler, decode_box_outputs

# COCO class ID for 'car'
CAR_CLASS_ID = 2  # Note: PyTorch COCO uses 0-indexed (car=2), TF uses 1-indexed (car=3)


class EfficientDetPyTorch:
    """
    PyTorch EfficientDet wrapper with pre-NMS access for adversarial attacks.

    Key Features:
    - Access to raw class logits (before sigmoid/NMS)
    - Access to raw box predictions (before NMS)
    - Differentiable forward pass (gradients flow properly)
    - Compatible with TensorFlow renderer via numpy bridge

    Usage:
        detector = EfficientDetPyTorch()
        class_logits, box_preds = detector.forward_pre_nms(images_torch)
        # Compute loss on raw logits (differentiable!)
    """

    def __init__(self, model_name='tf_efficientdet_d0', pretrained=True, device=None):
        """
        Initialize PyTorch EfficientDet model.

        Args:
            model_name: EfficientDet variant (d0-d7)
            pretrained: Load pretrained weights
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        print(f"Initializing PyTorch EfficientDet...")
        print(f"  Model: {model_name}")
        print(f"  Pretrained: {pretrained}")

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f"  Device: {self.device}")

        # Create model
        try:
            self.model = create_model(
                model_name,
                pretrained=pretrained,
                num_classes=90,  # COCO has 90 classes
                image_size=(512, 512),  # Input size for EfficientDet-D0
            )
            self.model.to(self.device)
            self.model.eval()  # Inference mode
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load EfficientDet model: {e}")

        # Get model config for anchor generation
        self.config = get_efficientdet_config(model_name)
        self.car_class_id = CAR_CLASS_ID

        # Initialize anchors
        self.anchors = Anchors.from_config(self.config)
        self.num_classes = 90
        self.num_anchors_per_loc = len(self.config.aspect_ratios) * self.config.num_scales  # 9

        # ImageNet normalization (EfficientDet expects normalized input)
        self.register_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.register_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        print()

    def preprocess(self, images_np):
        """
        Preprocess images for EfficientDet.

        Args:
            images_np: NumPy array [batch, H, W, 3] float32 [0, 1] (NHWC from TF)

        Returns:
            images_torch: PyTorch tensor [batch, 3, H, W] float32 [0, 1] (NCHW)
        """
        # Convert to PyTorch tensor
        images_torch = torch.from_numpy(images_np).to(self.device)

        # NHWC → NCHW
        images_torch = images_torch.permute(0, 3, 1, 2)

        # Resize to 512×512 if needed
        if images_torch.shape[2:] != (512, 512):
            images_torch = F.interpolate(
                images_torch,
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            )

        return images_torch

    def forward_pre_nms(self, images_torch):
        """
        Forward pass returning PRE-NMS outputs (NO gradients).

        This method is for inference/evaluation only. For training with
        gradient flow, use forward_pre_nms_with_grad() instead.

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1]

        Returns:
            class_logits: [batch, num_anchors, num_classes] raw logits
            box_preds: [batch, num_anchors, 4] box deltas
        """
        with torch.no_grad():
            return self._forward_pre_nms_impl(images_torch)

    def forward_pre_nms_with_grad(self, images_torch):
        """
        Forward pass returning PRE-NMS outputs WITH gradient tracking.

        This is the key method for adversarial attacks with end-to-end
        gradient flow. Unlike forward_pre_nms(), this method does NOT
        wrap the computation in torch.no_grad(), allowing gradients to
        flow back through the detector to the renderer and texture.

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1]
                         Should have requires_grad=True somewhere upstream

        Returns:
            class_logits: [batch, num_anchors, num_classes] raw logits (with grad)
            box_preds: [batch, num_anchors, 4] box deltas (with grad)
        """
        return self._forward_pre_nms_impl(images_torch)

    def _forward_pre_nms_impl(self, images_torch):
        """
        Internal implementation of pre-NMS forward pass.

        Shared by both forward_pre_nms() and forward_pre_nms_with_grad().

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1]

        Returns:
            class_logits: [batch, num_anchors, num_classes] raw logits
            box_preds: [batch, num_anchors, 4] box deltas
        """
        # Apply ImageNet normalization (differentiable)
        images_torch = (images_torch - self.register_mean) / self.register_std

        # Run model forward pass
        # EfficientDet returns (class_out_list, box_out_list) — one per FPN level
        outputs = self.model(images_torch)

        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            class_out, box_out = outputs

            A = self.num_anchors_per_loc  # 9
            NC = self.num_classes          # 90

            # FPN class outputs: list of [B, A*NC, H, W] tensors
            # Reshape each to [B, H*W*A, NC] by deinterleaving anchors from classes
            if isinstance(class_out, list):
                class_out_flat = []
                for c in class_out:
                    B, _, H, W = c.shape
                    # [B, A*NC, H, W] → [B, H, W, A*NC] → [B, H, W, A, NC] → [B, H*W*A, NC]
                    c = c.permute(0, 2, 3, 1).reshape(B, H, W, A, NC).reshape(B, H * W * A, NC)
                    class_out_flat.append(c)
                class_out = torch.cat(class_out_flat, dim=1)  # [B, total_anchors, 90]

            # FPN box outputs: list of [B, A*4, H, W] tensors
            if isinstance(box_out, list):
                box_out_flat = []
                for b in box_out:
                    B, _, H, W = b.shape
                    # [B, A*4, H, W] → [B, H, W, A*4] → [B, H, W, A, 4] → [B, H*W*A, 4]
                    b = b.permute(0, 2, 3, 1).reshape(B, H, W, A, 4).reshape(B, H * W * A, 4)
                    box_out_flat.append(b)
                box_out = torch.cat(box_out_flat, dim=1)  # [B, total_anchors, 4]

        else:
            raise ValueError(f"Unexpected output format from effdet: {type(outputs)}")

        return class_out, box_out

    def _decode_boxes(self, box_preds):
        """
        Decode raw box deltas into absolute [x1, y1, x2, y2] coordinates.

        Args:
            box_preds: [batch, num_anchors, 4] raw box deltas

        Returns:
            decoded_boxes: [batch, num_anchors, 4] in [x1, y1, x2, y2] format
        """
        anchor_boxes = self.anchors.boxes.to(box_preds.device)  # [num_anchors, 4]
        batch_size = box_preds.shape[0]
        decoded = []
        for i in range(batch_size):
            decoded_i = decode_box_outputs(box_preds[i], anchor_boxes, output_xyxy=True)
            decoded.append(decoded_i)
        return torch.stack(decoded, dim=0)  # [B, num_anchors, 4]

    def detect_cars_only(self, images_np, score_threshold=0.5):
        """
        Run full detection pipeline with NMS (for evaluation, not training).

        Args:
            images_np: NumPy [batch, H, W, 3] float32 [0, 1]
            score_threshold: Confidence threshold

        Returns:
            boxes: List of [N, 4] arrays (x1, y1, x2, y2) (one per image)
            scores: List of [N] arrays
            classes: List of [N] arrays
        """
        images_torch = self.preprocess(images_np)

        with torch.no_grad():
            class_out, box_out = self._forward_pre_nms_impl(images_torch)

            # Decode box deltas to absolute coordinates
            decoded_boxes = self._decode_boxes(box_out)  # [B, anchors, 4] xyxy

            # Apply sigmoid to get probabilities
            class_probs = torch.sigmoid(class_out)  # [batch, anchors, classes]
            car_probs = class_probs[:, :, self.car_class_id]  # [batch, anchors]

            results_boxes = []
            results_scores = []
            results_classes = []

            batch_size = images_torch.shape[0]
            for i in range(batch_size):
                scores_i = car_probs[i]
                boxes_i = decoded_boxes[i]

                mask = scores_i > score_threshold
                filtered_scores = scores_i[mask]
                filtered_boxes = boxes_i[mask]

                if len(filtered_scores) > 0:
                    # Apply NMS to remove duplicate detections
                    nms_keep = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
                    filtered_boxes = filtered_boxes[nms_keep]
                    filtered_scores = filtered_scores[nms_keep]

                    top_k = min(100, len(filtered_scores))
                    results_boxes.append(filtered_boxes[:top_k].cpu().numpy())
                    results_scores.append(filtered_scores[:top_k].cpu().numpy())
                    results_classes.append(np.full(top_k, self.car_class_id))
                else:
                    results_boxes.append(np.zeros((0, 4)))
                    results_scores.append(np.zeros((0,)))
                    results_classes.append(np.zeros((0,)))

            return results_boxes, results_scores, results_classes

    def detect_cars_with_boxes(self, images_torch, score_threshold=0.01):
        """
        Detect cars and return decoded bounding boxes + scores (no grad).

        Designed for visualization: returns boxes in pixel coordinates
        relative to the 512x512 detector input.

        Args:
            images_torch: [batch, 3, 512, 512] float32 [0, 1] (NCHW)
            score_threshold: Confidence threshold (default 0.01 = 1%)

        Returns:
            List of dicts per image, each with:
                'boxes': np.ndarray [N, 4] (x1, y1, x2, y2) in 512x512 coords
                'scores': np.ndarray [N]
        """
        with torch.no_grad():
            class_out, box_out = self._forward_pre_nms_impl(images_torch)
            decoded_boxes = self._decode_boxes(box_out)
            car_probs = torch.sigmoid(class_out[:, :, self.car_class_id])

            results = []
            batch_size = images_torch.shape[0]
            for i in range(batch_size):
                scores_i = car_probs[i]
                boxes_i = decoded_boxes[i]

                mask = scores_i > score_threshold
                filtered_scores = scores_i[mask]
                filtered_boxes = boxes_i[mask]

                if len(filtered_scores) > 0:
                    # Apply NMS to remove duplicate detections
                    nms_keep = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
                    filtered_boxes = filtered_boxes[nms_keep]
                    filtered_scores = filtered_scores[nms_keep]

                    # Keep top-100 by score (already sorted by NMS)
                    top_k = min(100, len(filtered_scores))
                    results.append({
                        'boxes': filtered_boxes[:top_k].cpu().numpy(),
                        'scores': filtered_scores[:top_k].cpu().numpy(),
                    })
                else:
                    results.append({
                        'boxes': np.zeros((0, 4)),
                        'scores': np.zeros((0,)),
                    })

            return results

    def get_max_car_confidence(self, class_logits):
        """
        Get maximum car confidence from raw logits (for loss computation).

        Args:
            class_logits: [batch, anchors, classes] raw logits

        Returns:
            max_conf: [batch] maximum car confidence per image
        """
        # Get car logits
        car_logits = class_logits[:, :, self.car_class_id]  # [batch, anchors]

        # Apply sigmoid to get probabilities
        car_probs = torch.sigmoid(car_logits)

        # Get max confidence per image
        max_conf, _ = torch.max(car_probs, dim=1)  # [batch]

        return max_conf

    def get_detector_info(self):
        """Get detector metadata."""
        return {
            'framework': 'PyTorch',
            'model': self.config.name,
            'car_class_id': self.car_class_id,
            'device': str(self.device),
            'input_size': (512, 512),
            'differentiable': True,  # Pre-NMS outputs are differentiable!
        }


# Test script
if __name__ == "__main__":
    print("=" * 70)
    print("PYTORCH EFFICIENTDET TEST")
    print("=" * 70)
    print()

    # Initialize detector
    detector = EfficientDetPyTorch()

    # Test with random images (NumPy format from TensorFlow)
    print("Testing with batch of 2 random images (500×500)...")
    test_images_np = np.random.rand(2, 500, 500, 3).astype(np.float32)
    print(f"  Input shape (NumPy): {test_images_np.shape}")
    print()

    # Test preprocessing
    print("Testing preprocessing...")
    images_torch = detector.preprocess(test_images_np)
    print(f"  Output shape (PyTorch): {images_torch.shape}")
    print(f"  Device: {images_torch.device}")
    print()

    # Test pre-NMS forward pass
    print("Testing pre-NMS forward pass...")
    class_out, box_out = detector.forward_pre_nms(images_torch)
    print(f"  Class logits shape: {class_out.shape}")
    print(f"  Box predictions shape: {box_out.shape}")
    print()

    # Test max car confidence extraction
    print("Testing max car confidence...")
    max_conf = detector.get_max_car_confidence(class_out)
    print(f"  Max confidence shape: {max_conf.shape}")
    print(f"  Max confidence values: {max_conf.cpu().numpy()}")
    print()

    # Test full detection with NMS
    print("Testing full detection with NMS...")
    boxes, scores, classes = detector.detect_cars_only(test_images_np, score_threshold=0.3)
    print(f"  Image 0: {len(boxes[0])} cars detected")
    print(f"  Image 1: {len(boxes[1])} cars detected")
    print()

    # Show detector info
    print("Detector Information:")
    info = detector.get_detector_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 70)
    print("PYTORCH DETECTOR TEST COMPLETE ✓")
    print("=" * 70)
