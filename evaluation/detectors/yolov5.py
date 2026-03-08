"""YOLOv5 detector wrapper (loaded via torch.hub)."""

import sys
import os
import numpy as np
import torch

COCO_CAR_CLASS = 2  # 0-indexed


class YOLOv5Detector:
    """YOLOv5 wrapper matching the detect_cars_with_boxes interface."""

    def __init__(self, variant='yolov5s', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        print(f"Loading YOLOv5 ({variant}) ...")
        # Temporarily remove project root from sys.path and clear cached
        # 'models' module to prevent our models/ package from shadowing
        # YOLOv5's internal models/ module
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        removed_paths = []
        for p in list(sys.path):
            if os.path.realpath(p) == os.path.realpath(project_root):
                sys.path.remove(p)
                removed_paths.append(p)
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == 'models' or key.startswith('models.'):
                saved_modules[key] = sys.modules.pop(key)
        try:
            self.model = torch.hub.load(
                'ultralytics/yolov5', variant, pretrained=True, verbose=False,
            )
        finally:
            # Remove YOLOv5's models from sys.modules, restore ours
            for key in list(sys.modules.keys()):
                if key == 'models' or key.startswith('models.'):
                    del sys.modules[key]
            sys.modules.update(saved_modules)
            for p in reversed(removed_paths):
                sys.path.insert(0, p)
        self.model.to(self.device)
        self.model.eval()
        self.variant = variant
        self.det_size = 640
        print(f"  YOLOv5 ({variant}) loaded on {self.device}")

    def detect_cars_with_boxes(self, images_torch, score_threshold=0.01):
        """Run YOLOv5 and return car detections.

        Args:
            images_torch: [B, 3, H, W] float32 tensor in [0, 1]
            score_threshold: minimum confidence

        Returns:
            list of dicts per image with 'boxes' [N,4] and 'scores' [N]
            Boxes are in the input tensor's coordinate space.
        """
        B, _, H, W = images_torch.shape
        results_list = []

        for i in range(B):
            img_np = (images_torch[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            with torch.no_grad():
                results = self.model(img_np, size=max(H, W))

            dets = results.xyxy[0].cpu().numpy()

            if len(dets) == 0:
                results_list.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'scores': np.zeros((0,), dtype=np.float32),
                })
                continue

            car_mask = dets[:, 5].astype(int) == COCO_CAR_CLASS
            dets = dets[car_mask]

            if len(dets) == 0:
                results_list.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'scores': np.zeros((0,), dtype=np.float32),
                })
                continue

            scores = dets[:, 4].astype(np.float32)
            boxes = dets[:, :4].astype(np.float32)

            keep = scores >= score_threshold
            scores = scores[keep]
            boxes = boxes[keep]

            order = np.argsort(-scores)
            results_list.append({
                'boxes': boxes[order],
                'scores': scores[order],
            })

        return results_list
