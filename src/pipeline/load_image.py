from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}


def _load_standard_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def _load_dicom(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path))
    data = dataset.pixel_array.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return data * slope + intercept


def load_image(path: Path | str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".dcm":
        return _load_dicom(image_path)

    return _load_standard_image(image_path)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        rgb = image.astype(np.float32)
        image = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        image = image.astype(np.float32)

    min_val = float(image.min())
    max_val = float(image.max())
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = image * 0.0

    return image[None, ...]
