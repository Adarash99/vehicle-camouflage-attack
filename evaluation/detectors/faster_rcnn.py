"""Faster R-CNN detector (torchvision, ResNet50-FPN backbone, COCO pretrained)."""

from .torchvision_base import TorchvisionDetector


class FasterRCNNDetector(TorchvisionDetector):
    def __init__(self, device=None):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        print("Loading Faster R-CNN (ResNet50-FPN) ...")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        super().__init__(model, det_size=800, name='Faster-RCNN-R50-FPN', device=device)
