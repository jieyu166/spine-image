"""Vertebra segmentation and landmark inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .vertebra_types import VertebraDetection, VertebraResult


@dataclass
class VertebraDetectorConfig:
    """Configuration for loading vertebra detectors."""

    model_path: Optional[str] = None
    device: str = "cpu"


class VertebraDetector:
    """Model loader and inference wrapper for vertebra detection."""

    def __init__(self, config: VertebraDetectorConfig) -> None:
        self._config = config

    @classmethod
    def load(cls, model_path: Optional[str] = None, device: str = "cpu") -> "VertebraDetector":
        """Load a detector from a checkpoint path.

        This method is intended to be extended with actual model loading logic.
        """

        config = VertebraDetectorConfig(model_path=model_path, device=device)
        return cls(config)

    def predict(self, image: Any, image_id: Optional[str] = None) -> VertebraResult:
        """Run vertebra detection on an image.

        Args:
            image: Image array or tensor input.
            image_id: Optional identifier for the image.

        Returns:
            VertebraResult containing detections with contours or keypoints.
        """

        detections = self._predict_detections(image)
        return VertebraResult(image_id=image_id, detections=detections)

    def _predict_detections(self, image: Any) -> Sequence[VertebraDetection]:
        """Internal inference hook.

        Override this method when wiring up a real model backend.
        """

        return []
