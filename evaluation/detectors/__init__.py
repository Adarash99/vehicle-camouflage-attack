"""Detector wrappers for adversarial texture evaluation.

All detectors implement the same interface:
    detector.detect_cars_with_boxes(images_torch, score_threshold=0.01)
        images_torch: [B, 3, H, W] float32 tensor in [0, 1]
        returns: list of dicts per image with 'boxes' [N,4] and 'scores' [N]
"""

SUPPORTED_DETECTORS = [
    'efficientdet', 'yolov5s', 'yolov5m', 'yolov5l',
    'ssd', 'faster_rcnn', 'mask_rcnn',
]


def create_detector(name):
    """Create a detector by name.

    Returns:
        (detector, det_size) -- detector instance and its native input resolution.
    """
    if name == 'efficientdet':
        from .efficientdet import EfficientDetWrapper
        return EfficientDetWrapper(), 512

    elif name.startswith('yolov5'):
        from .yolov5 import YOLOv5Detector
        return YOLOv5Detector(variant=name), 640

    elif name == 'ssd':
        from .ssd import SSDDetector
        return SSDDetector(), 300

    elif name == 'faster_rcnn':
        from .faster_rcnn import FasterRCNNDetector
        return FasterRCNNDetector(), 800

    elif name == 'mask_rcnn':
        from .mask_rcnn import MaskRCNNDetector
        return MaskRCNNDetector(), 800

    else:
        raise ValueError(
            f"Unknown detector: {name}. "
            f"Supported: {', '.join(SUPPORTED_DETECTORS)}"
        )
