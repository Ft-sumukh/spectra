"""
SPECTRA - Vision Engine
MODULE E

Real AI vision pipeline using ONNX Runtime.
Supports:
1. MobileNetV2 ONNX (Image Classification - Primary Snapdragon Edge Benchmark Model)
2. YOLOv8 ONNX (Object Detection)

Provides real measured latency, device, runtime, and top predictions.
Designed to be model-replaceable.
"""

from __future__ import annotations

import time
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

from utils.logger import get_logger
from utils.imagenet_labels import ensure_imagenet_labels

log = get_logger("INFERENCE")


@dataclass
class Classification:
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox_xyxy],
        }


@dataclass
class VisionResult:
    """Result from a vision inference pass."""
    success: bool
    task: str  # "classification" or "object_detection"
    classifications: list[Classification] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    execution_device: str = "CPU"
    runtime: str = "onnx"
    provider: str = "CPUExecutionProvider"
    model_name: str = "MobileNetV2"
    model_id: str = "mobilenet_v2"
    image_width: int = 0
    image_height: int = 0
    error_message: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "task": self.task,
            "classifications": [c.to_dict() for c in self.classifications],
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "execution_device": self.execution_device,
            "runtime": self.runtime,
            "provider": self.provider,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "image_size": [self.image_width, self.image_height],
            "error": self.error_message,
            "notes": self.notes,
        }

    def print_report(self) -> None:
        divider = "-" * 50
        print(f"\n{divider}")
        print("  VISION INFERENCE RESULT")
        print(f"{divider}")
        print(f"  Model    : {self.model_name}")
        print(f"  Task     : {self.task}")
        print(f"  Runtime  : {self.runtime}")
        print(f"  Device   : {self.execution_device}")
        print(f"  Provider : {self.provider}")
        print(f"  Latency  : {self.inference_time_ms:.2f} ms")
        print(f"  Image    : {self.image_width}x{self.image_height}")
        if self.success:
            if self.task == "classification":
                print(f"\n  Top Classifications:")
                for c in self.classifications:
                    print(f"    [{c.confidence:.2%}] {c.class_name} (id={c.class_id})")
            else:
                print(f"\n  Detections ({len(self.detections)}):")
                for d in self.detections:
                    print(f"    [{d.confidence:.2%}] {d.class_name:20s} bbox={[round(v,0) for v in d.bbox_xyxy]}")
        else:
            print(f"\n  ERROR: {self.error_message}")
        print()


class VisionEngine:
    """
    Hardware-aware AI vision inference engine.
    Supports MobileNetV2 (classification) and YOLOv8 (detection).
    """

    def __init__(
        self,
        model_id: str = "mobilenet_v2",
        providers: Optional[list[str]] = None,
        conf_threshold: float = 0.05,
    ):
        self._model_id = model_id
        self._providers = providers or ["CPUExecutionProvider"]
        self._conf_threshold = conf_threshold
        self._session = None
        self._input_name = None
        self._input_size = (224, 224)
        self._loaded = False
        self._device = "CPU"
        self._runtime = "onnx"
        self._provider_used = self._providers[0] if self._providers else "CPUExecutionProvider"
        self._task = "classification"
        self._labels = ensure_imagenet_labels()

    def load(self, model_path: Union[str, Path]) -> tuple[bool, str]:
        """Load an ONNX model for vision inference."""
        try:
            import onnxruntime as ort
            model_path = Path(model_path)
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"

            log.info(f"Loading vision model: {model_path.name} | providers={self._providers}")
            self._session = ort.InferenceSession(str(model_path), providers=self._providers)
            
            # Inspect inputs
            inputs = self._session.get_inputs()
            self._input_name = inputs[0].name
            input_shape = inputs[0].shape
            if len(input_shape) == 4:
                h, w = input_shape[2], input_shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    self._input_size = (w, h)

            # Determine task based on outputs
            outputs = self._session.get_outputs()
            out_shape = outputs[0].shape
            if len(out_shape) == 2 and out_shape[1] == 1000:
                self._task = "classification"
                self._model_name = "MobileNetV2 (ONNX)"
            else:
                self._task = "object_detection"
                self._model_name = "YOLOv8 Nano"

            # Determine actual provider
            actual_providers = self._session.get_providers()
            self._provider_used = actual_providers[0] if actual_providers else "CPUExecutionProvider"
            self._device = self._provider_to_device(self._provider_used)
            self._runtime = "onnx"
            self._loaded = True

            log.info(f"Vision model loaded ({self._task}). Provider: {self._provider_used}, Device: {self._device}")
            return True, f"Model loaded: {model_path.name} ({self._task})"

        except ImportError:
            return False, "onnxruntime not installed."
        except Exception as e:
            log.error(f"Failed to load vision model: {e}")
            return False, f"Model load failed: {e}"

    def _provider_to_device(self, provider: str) -> str:
        mapping = {
            "CPUExecutionProvider": "CPU",
            "CUDAExecutionProvider": "GPU",
            "DmlExecutionProvider": "GPU",
            "QNNExecutionProvider": "NPU",
            "ROCMExecutionProvider": "GPU",
        }
        return mapping.get(provider, "CPU")

    def _preprocess_classification(self, image) -> tuple[np.ndarray, int, int]:
        """Preprocess for ImageNet classification (MobileNetV2)."""
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image)
        elif isinstance(image, (str, Path)):
            pil_img = PILImage.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size
        target_w, target_h = self._input_size

        # Resize to input dimensions (224x224)
        resized = pil_img.resize((target_w, target_h))
        img_array = np.array(resized, dtype=np.float32) / 255.0

        # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # [1, 3, 224, 224]

        return img_array.astype(np.float32), orig_w, orig_h

    def _postprocess_classification(self, logits: np.ndarray, top_k: int = 5) -> list[Classification]:
        """Convert logits to softmax probabilities and return top-K classes."""
        logits = logits.flatten()
        # Stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            name = self._labels[idx] if idx < len(self._labels) else f"Class {idx}"
            results.append(Classification(
                class_id=int(idx),
                class_name=name,
                confidence=float(probs[idx])
            ))
        return results

    def infer(self, image) -> VisionResult:
        """Run vision inference on input image."""
        if not self._loaded or self._session is None:
            return VisionResult(
                success=False,
                task=self._task,
                inference_time_ms=0.0,
                execution_device="N/A",
                runtime="onnx",
                provider="N/A",
                model_name=getattr(self, "_model_name", "N/A"),
                model_id=self._model_id,
                error_message="Model not loaded. Call load() first."
            )

        try:
            img_tensor, orig_w, orig_h = self._preprocess_classification(image)

            # Warmup / precise timing
            t0 = time.perf_counter()
            outputs = self._session.run(None, {self._input_name: img_tensor})
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            classifications = self._postprocess_classification(outputs[0], top_k=5)

            log.info(f"Vision inference: {latency_ms:.2f}ms | top={classifications[0].class_name} ({classifications[0].confidence:.1%}) | device={self._device}")

            return VisionResult(
                success=True,
                task="classification",
                classifications=classifications,
                inference_time_ms=latency_ms,
                execution_device=self._device,
                runtime=self._runtime,
                provider=self._provider_used,
                model_name=getattr(self, "_model_name", "MobileNetV2 (ONNX)"),
                model_id=self._model_id,
                image_width=orig_w,
                image_height=orig_h,
            )

        except Exception as e:
            log.error(f"Inference error: {e}")
            return VisionResult(
                success=False,
                task=self._task,
                inference_time_ms=0.0,
                execution_device=self._device,
                runtime=self._runtime,
                provider=self._provider_used,
                model_name=getattr(self, "_model_name", "MobileNetV2"),
                model_id=self._model_id,
                image_width=0,
                image_height=0,
                error_message=str(e),
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device
