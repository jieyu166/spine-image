"""Data structures for vertebra segmentation and landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class VertebraDetection:
    """Single vertebra detection output.

    Attributes:
        vertebra_id: Vertebra label identifier (e.g., "L1", "T12").
        contour: Ordered boundary points for segmentation output.
        keypoints: Named landmark points for the vertebra (if available).
        score: Optional confidence score for the detection.
    """

    vertebra_id: str
    contour: Optional[Sequence[Point]] = None
    keypoints: Optional[Mapping[str, Point]] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class VertebraResult:
    """Collection of vertebra detections for a single image."""

    image_id: Optional[str]
    detections: Sequence[VertebraDetection]
