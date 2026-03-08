"""EfficientDet-D0 detector (same as the training target)."""

from attack.detector_pytorch import EfficientDetPyTorch


class EfficientDetWrapper(EfficientDetPyTorch):
    """Thin wrapper so all detectors live under evaluation.detectors."""
    pass
