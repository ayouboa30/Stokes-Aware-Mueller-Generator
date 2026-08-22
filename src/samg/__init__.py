"""Stokes-Aware Mueller Generator research package."""

from .models import (
    CNNConditionalPIVAE,
    CNNVanillaVAE,
    ConditionalCNNPIVAE,
    DirectMuellerVAE,
    FourIncidentPIVAE,
    OperatorPIVAE,
    VanillaVAE,
)
from .operator import operator_consistency_metrics, operator_loss_bundle

__all__ = [
    "CNNConditionalPIVAE",
    "CNNVanillaVAE",
    "ConditionalCNNPIVAE",
    "DirectMuellerVAE",
    "FourIncidentPIVAE",
    "OperatorPIVAE",
    "VanillaVAE",
    "operator_consistency_metrics",
    "operator_loss_bundle",
]

__version__ = "0.1.0"
