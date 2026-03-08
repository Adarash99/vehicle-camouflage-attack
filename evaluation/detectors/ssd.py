"""SSD300 detector (torchvision, VGG16 backbone, COCO pretrained)."""

from .torchvision_base import TorchvisionDetector


class SSDDetector(TorchvisionDetector):
    def __init__(self, device=None):
        from torchvision.models.detection import ssd300_vgg16
        print("Loading SSD300-VGG16 ...")
        model = ssd300_vgg16(pretrained=True)
        super().__init__(model, det_size=300, name='SSD300-VGG16', device=device)
