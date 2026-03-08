"""Base class for torchvision detection model wrappers."""

import numpy as np
import torch

# COCO 1-indexed class IDs (torchvision convention)
COCO_CAR_CLASS_1IDX = 3


class TorchvisionDetector:
    """Shared inference logic for torchvision detection models.

    Subclasses only need to supply the loaded model, det_size, and name.
    """

    def __init__(self, model, det_size, name, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.det_size = det_size
        self.name = name
        print(f"  {name} loaded on {self.device}")

    def detect_cars_with_boxes(self, images_torch, score_threshold=0.01):
        """Run detector and return car detections.

        Args:
            images_torch: [B, 3, H, W] float32 tensor in [0, 1]
            score_threshold: minimum confidence

        Returns:
            list of dicts per image with 'boxes' [N,4] and 'scores' [N]
            Boxes are in the input tensor's coordinate space.
        """
        B = images_torch.shape[0]
        images_list = [images_torch[i].to(self.device) for i in range(B)]

        with torch.no_grad():
            outputs = self.model(images_list)

        results_list = []
        for output in outputs:
            labels = output['labels'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy().astype(np.float32)
            scores = output['scores'].cpu().numpy().astype(np.float32)

            # Filter for car class only
            car_mask = labels == COCO_CAR_CLASS_1IDX
            boxes = boxes[car_mask]
            scores = scores[car_mask]

            if len(scores) == 0:
                results_list.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'scores': np.zeros((0,), dtype=np.float32),
                })
                continue

            # Threshold
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]

            # Sort by confidence descending
            order = np.argsort(-scores)
            results_list.append({
                'boxes': boxes[order],
                'scores': scores[order],
            })

        return results_list
