"""Mask R-CNN detector (torchvision, ResNet50-FPN backbone, COCO pretrained)."""

from .torchvision_base import TorchvisionDetector


class MaskRCNNDetector(TorchvisionDetector):
    def __init__(self, device=None):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        print("Loading Mask R-CNN (ResNet50-FPN) ...")
        model = maskrcnn_resnet50_fpn(pretrained=True)
        super().__init__(model, det_size=800, name='Mask-RCNN-R50-FPN', device=device)
