"""Downstream heads and polarimetrically coherent augmentation."""

from .augmentation import apply_mueller_reference_change, polarimetric_d4
from .multitask import ClassificationHead, MultiTaskModel, SegmentationHead, SharedTrunk

__all__ = [
    "ClassificationHead",
    "MultiTaskModel",
    "SegmentationHead",
    "SharedTrunk",
    "apply_mueller_reference_change",
    "polarimetric_d4",
]
