"""Pipeline for vertebra detection."""

from __future__ import annotations

from typing import Any, Optional

from src.models.vertebra_detector import VertebraDetector
from src.models.vertebra_types import VertebraResult


def detect_vertebrae(
    image: Any,
    detector: VertebraDetector,
    image_id: Optional[str] = None,
) -> VertebraResult:
    """Detect vertebrae and return contours or landmarks.

    Args:
        image: Image array or tensor input.
        detector: Loaded vertebra detector.
        image_id: Optional identifier for the image.

    Returns:
        VertebraResult containing per-vertebra detections.
    """

    return detector.predict(image, image_id=image_id)
