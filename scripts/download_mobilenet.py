"""
Download and verify MobileNetV2 ONNX model for SPECTRA.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FILE = MODELS_DIR / "mobilenetv2-7.onnx"

# URLs to try (reliable ONNX model zoo mirrors)
URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    "https://huggingface.co/datasets/Xenova/transformers.js-models/resolve/main/mobilenet_v2_1.0_224.onnx",
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
]

def download_mobilenetv2():
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1000000:
        print(f"[MODEL] MobileNetV2 already exists at {TARGET_FILE} ({TARGET_FILE.stat().st_size / 1024 / 1024:.2f} MB)")
        return True

    print(f"[MODEL] Downloading MobileNetV2 ONNX to {TARGET_FILE}...")
    for url in URLS:
        try:
            print(f"[MODEL] Attempting download from: {url}")
            urllib.request.urlretrieve(url, TARGET_FILE)
            if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1000000:
                print(f"[MODEL] Successfully downloaded ({TARGET_FILE.stat().st_size / 1024 / 1024:.2f} MB)")
                return True
        except Exception as e:
            print(f"[MODEL] Download from {url} failed: {e}")

    # Fallback: export via torch if available
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print("[MODEL] Attempting export via transformers / torch...")
        model_name = "google/mobilenet_v2_1.0_224"
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            str(TARGET_FILE),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14
        )
        if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1000000:
            print(f"[MODEL] Exported MobileNetV2 ONNX ({TARGET_FILE.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
    except Exception as e:
        print(f"[MODEL] Torch export fallback failed: {e}")

    return False

if __name__ == "__main__":
    success = download_mobilenetv2()
    if not success:
        sys.exit(1)
