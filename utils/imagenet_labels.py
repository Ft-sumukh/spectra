"""
ImageNet-1K class labels for MobileNetV2 classification.
"""

import urllib.request
import json
from pathlib import Path

LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "imagenet_classes.txt"

def ensure_imagenet_labels() -> list[str]:
    """Ensure ImageNet class labels exist and return them as a list."""
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        urllib.request.urlretrieve(url, LABELS_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        # Fallback minimal labels if offline
        return [f"class_{i}" for i in range(1000)]
