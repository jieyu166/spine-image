"""Model interfaces for spine image processing."""

from .vertebra_detector import VertebraDetector
from .vertebra_types import VertebraDetection, VertebraResult

__all__ = ["VertebraDetector", "VertebraDetection", "VertebraResult"]
