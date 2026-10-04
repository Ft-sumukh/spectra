"""
SPECTRA - Model Manager
MODULE C

Manages AI model registration, loading, unloading, and compatibility checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("MODEL")


class Modality(str, Enum):
    VISION = "vision"
    TEXT = "text"
    SPEECH = "speech"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


class ModelTask(str, Enum):
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    OCR = "ocr"
    SUMMARIZATION = "summarization"
    QA = "question_answering"
    SPEECH_RECOGNITION = "speech_recognition"
    EMBEDDING = "embedding"
    GENERATION = "generation"
    SEGMENTATION = "segmentation"


class ModelFormat(str, Enum):
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    GGUF = "gguf"
    OPENVINO = "openvino"


class ModelStatus(str, Enum):
    REGISTERED = "registered"    # Known but not downloaded
    DOWNLOADED = "downloaded"    # Downloaded but not loaded into memory
    LOADED = "loaded"            # In memory, ready for inference
    ERROR = "error"              # Failed to load
    INCOMPATIBLE = "incompatible"


@dataclass
class ModelSpec:
    """Specification and metadata for an AI model."""
    model_id: str
    name: str
    version: str
    modality: Modality
    task: ModelTask
    format: ModelFormat
    quantization: Optional[str]          # "fp32", "fp16", "int8", "int4", None
    supported_runtimes: list[str]        # ["onnx_cpu", "onnx_gpu", "onnx_qnn"]
    supported_devices: list[str]         # ["CPU", "GPU", "NPU"]
    memory_requirement_mb: float
    input_type: str                      # "image", "text", "audio", "document"
    output_type: str                     # "bounding_boxes", "text", "embedding", etc.
    local_only: bool = True
    model_path: Optional[str] = None     # Relative to MODEL_DIRECTORY
    download_url: Optional[str] = None
    license: str = "unknown"
    description: str = ""
    status: ModelStatus = ModelStatus.REGISTERED
    load_error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert enums to strings for serialization
        d["modality"] = self.modality.value
        d["task"] = self.task.value
        d["format"] = self.format.value
        d["status"] = self.status.value
        return d


class ModelRegistry:
    """
    Central registry of all known models.
    Handles registration, discovery, and persistence.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._models: dict[str, ModelSpec] = {}
        self._registry_path = registry_path
        self._load_registry()

    def _load_registry(self) -> None:
        if self._registry_path and self._registry_path.exists():
            try:
                with open(self._registry_path, "r") as f:
                    data = json.load(f)
                for model_data in data.get("models", []):
                    try:
                        spec = self._dict_to_spec(model_data)
                        self._models[spec.model_id] = spec
                    except Exception as e:
                        log.error(f"Failed to load model spec: {e}")
                log.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                log.error(f"Failed to load model registry: {e}")

    def _save_registry(self) -> None:
        if self._registry_path:
            try:
                data = {"models": [m.to_dict() for m in self._models.values()]}
                with open(self._registry_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                log.error(f"Failed to save model registry: {e}")

    def _dict_to_spec(self, d: dict) -> ModelSpec:
        return ModelSpec(
            model_id=d["model_id"],
            name=d["name"],
            version=d["version"],
            modality=Modality(d["modality"]),
            task=ModelTask(d["task"]),
            format=ModelFormat(d["format"]),
            quantization=d.get("quantization"),
            supported_runtimes=d.get("supported_runtimes", ["onnx_cpu"]),
            supported_devices=d.get("supported_devices", ["CPU"]),
            memory_requirement_mb=d.get("memory_requirement_mb", 100.0),
            input_type=d.get("input_type", "unknown"),
            output_type=d.get("output_type", "unknown"),
            local_only=d.get("local_only", True),
            model_path=d.get("model_path"),
            download_url=d.get("download_url"),
            license=d.get("license", "unknown"),
            description=d.get("description", ""),
            status=ModelStatus(d.get("status", ModelStatus.REGISTERED.value)),
            load_error=d.get("load_error"),
        )

    def register(self, spec: ModelSpec) -> None:
        self._models[spec.model_id] = spec
        log.info(f"Model registered: {spec.model_id} ({spec.name})")
        self._save_registry()

    def get(self, model_id: str) -> Optional[ModelSpec]:
        return self._models.get(model_id)

    def list_models(self) -> list[ModelSpec]:
        return list(self._models.values())

    def list_by_task(self, task: ModelTask) -> list[ModelSpec]:
        return [m for m in self._models.values() if m.task == task]

    def list_by_modality(self, modality: Modality) -> list[ModelSpec]:
        return [m for m in self._models.values() if m.modality == modality]


class ModelManager:
    """
    Manages model lifecycle: registration, loading, unloading, inference readiness.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        from config import settings
        self._model_dir = model_dir or settings.MODEL_DIRECTORY
        registry_path = self._model_dir / "registry.json"
        self._registry = ModelRegistry(registry_path)
        self._loaded_models: dict[str, Any] = {}  # model_id -> loaded object
        self._register_builtin_models()

    def _register_builtin_models(self) -> None:
        """Register lightweight built-in models available for immediate use."""

        # MobileNetV2 ONNX — Snapdragon edge-optimized image classification
        self._registry.register(ModelSpec(
            model_id="mobilenet_v2",
            name="MobileNetV2 (ONNX)",
            version="2.0",
            modality=Modality.VISION,
            task=ModelTask.IMAGE_CLASSIFICATION,
            format=ModelFormat.ONNX,
            quantization="fp32",
            supported_runtimes=["onnx_cpu", "onnx_gpu", "onnx_qnn"],
            supported_devices=["CPU", "GPU", "NPU"],
            memory_requirement_mb=14.0,
            input_type="image",
            output_type="classification",
            local_only=True,
            model_path="mobilenetv2-7.onnx",
            download_url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            license="Apache-2.0",
            description="MobileNetV2 — Google edge-optimized CNN for image classification & Snapdragon NPU acceleration. 3.5M params.",
        ))

        # YOLOv10n ONNX — ultra-lightweight object detection.
        # Replaces the previous entry, which pointed at a .pt PyTorch
        # checkpoint that no ONNX loader could ever open.
        self._registry.register(ModelSpec(
            model_id="yolov8n_onnx",
            name="YOLOv10 Nano",
            version="10.0",
            modality=Modality.VISION,
            task=ModelTask.OBJECT_DETECTION,
            format=ModelFormat.ONNX,
            quantization="fp32",
            supported_runtimes=["onnx_cpu", "onnx_gpu", "onnx_qnn"],
            supported_devices=["CPU", "GPU", "NPU"],
            memory_requirement_mb=9.0,
            input_type="image",
            output_type="bounding_boxes",
            local_only=True,
            model_path="yolov10n.onnx",
            download_url=(
                "https://huggingface.co/onnx-community/yolov10n/"
                "resolve/main/onnx/model.onnx"
            ),
            license="AGPL-3.0",
            description=(
                "YOLOv10 Nano — 2.3M params, 9 MB. End-to-end NMS export "
                "(output [1,300,6]). The engine also decodes raw-head "
                "[1,84,8400] exports from YOLOv8. Fetch with "
                "scripts/verify_detection.py."
            ),
        ))

        # Sentence transformer for text embedding — lightweight
        self._registry.register(ModelSpec(
            model_id="sentence_transformer_mini",
            name="all-MiniLM-L6-v2",
            version="1.0",
            modality=Modality.TEXT,
            task=ModelTask.EMBEDDING,
            format=ModelFormat.PYTORCH,
            quantization=None,
            supported_runtimes=["pytorch_cpu"],
            supported_devices=["CPU"],
            memory_requirement_mb=90.0,
            input_type="text",
            output_type="embedding",
            local_only=True,
            description="Lightweight sentence embedding model. 22M params.",
        ))

        # Whisper tiny for speech recognition
        self._registry.register(ModelSpec(
            model_id="whisper_tiny",
            name="Whisper Tiny",
            version="v3",
            modality=Modality.SPEECH,
            task=ModelTask.SPEECH_RECOGNITION,
            format=ModelFormat.PYTORCH,
            quantization=None,
            supported_runtimes=["pytorch_cpu"],
            supported_devices=["CPU"],
            memory_requirement_mb=150.0,
            input_type="audio",
            output_type="text",
            local_only=True,
            description="OpenAI Whisper Tiny — local speech-to-text. 39M params.",
        ))

        # Qwen3-1.7B INT4 — default local text generation via ONNX Runtime GenAI.
        # Registered as a real GENERATION model so the router can select it.
        self._registry.register(ModelSpec(
            model_id="qwen3_1_7b_int4",
            name="Qwen3-1.7B (INT4, KLD block-128)",
            version="1.0",
            modality=Modality.TEXT,
            task=ModelTask.GENERATION,
            format=ModelFormat.ONNX,
            quantization="int4 (KLD, block 128)",
            supported_runtimes=["onnx_cpu", "onnx_gpu", "onnx_qnn"],
            supported_devices=["CPU", "GPU", "NPU"],
            memory_requirement_mb=1360.0,
            input_type="text",
            output_type="text",
            local_only=True,
            model_path="llm/qwen3_1_7b_int4/model.onnx",
            download_url=(
                "https://huggingface.co/onnx-community/Qwen3-1.7B-ONNX/"
                "resolve/main/onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128/"
            ),
            license="Apache-2.0",
            description=(
                "Qwen3-1.7B INT4 — default SPECTRA assistant. 1.7B params, "
                "1.35 GB on disk. Runs via ONNX Runtime GenAI, the same "
                "runtime that exposes the Qualcomm QNN provider, so CPU and "
                "NPU execution share one code path. Fetch with "
                "scripts/download_llm.py."
            ),
        ))

        # Qwen3-0.6B INT4 — low-memory alternative for constrained machines.
        self._registry.register(ModelSpec(
            model_id="qwen3_0_6b_int4",
            name="Qwen3-0.6B (INT4, KLD block-128)",
            version="1.0",
            modality=Modality.TEXT,
            task=ModelTask.GENERATION,
            format=ModelFormat.ONNX,
            quantization="int4 (KLD, block 128)",
            supported_runtimes=["onnx_cpu", "onnx_gpu", "onnx_qnn"],
            supported_devices=["CPU", "GPU", "NPU"],
            memory_requirement_mb=520.0,
            input_type="text",
            output_type="text",
            local_only=True,
            model_path="llm/qwen3_0_6b_int4/model.onnx",
            download_url=(
                "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/"
                "resolve/main/onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128/"
            ),
            license="Apache-2.0",
            description=(
                "Qwen3-0.6B INT4 — 511 MB, for machines that cannot spare "
                "1.35 GB. Answers questions about its own hardware "
                "unreliably; prefer the 1.7B model when memory allows."
            ),
        ))

        log.info(f"Built-in models registered: {len(self._registry.list_models())}")

    def register_model(self, spec: ModelSpec) -> None:
        self._registry.register(spec)

    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        return self._registry.get(model_id)

    def list_models(self) -> list[ModelSpec]:
        return self._registry.list_models()

    def check_compatibility(self, model_id: str, device: str, runtime: str) -> tuple[bool, str]:
        """Check if a model can run on the given device/runtime."""
        spec = self._registry.get(model_id)
        if not spec:
            return False, f"Model '{model_id}' not found in registry"
        if device not in spec.supported_devices:
            return False, f"Model '{spec.name}' does not support device '{device}'. Supported: {spec.supported_devices}"
        if runtime not in spec.supported_runtimes:
            return False, f"Model '{spec.name}' does not support runtime '{runtime}'. Supported: {spec.supported_runtimes}"
        path = self._model_dir / (spec.model_path or "")
        if spec.model_path and not path.exists():
            return False, f"Model file not found: {path}"
        return True, "Compatible"

    def is_model_file_present(self, model_id: str) -> bool:
        spec = self._registry.get(model_id)
        if not spec or not spec.model_path:
            return False
        return (self._model_dir / spec.model_path).exists()

    def load_model(self, model_id: str) -> tuple[bool, str]:
        """Load a model into memory. Returns (success, message)."""
        spec = self._registry.get(model_id)
        if not spec:
            return False, f"Model '{model_id}' not registered"

        if model_id in self._loaded_models:
            return True, f"Model '{model_id}' already loaded"

        log.info(f"Loading model: {model_id} ({spec.name})")

        if spec.format == ModelFormat.ONNX:
            return self._load_onnx_model(model_id, spec)
        elif spec.format == ModelFormat.PYTORCH:
            return self._load_pytorch_model(model_id, spec)
        else:
            return False, f"Format '{spec.format}' loader not yet implemented"

    def _load_onnx_model(self, model_id: str, spec: ModelSpec) -> tuple[bool, str]:
        try:
            import onnxruntime as ort
            path = self._model_dir / (spec.model_path or "")
            if not path.exists():
                spec.status = ModelStatus.REGISTERED
                return False, f"Model file not found: {path}. Please download the model first."
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._loaded_models[model_id] = sess
            spec.status = ModelStatus.LOADED
            log.info(f"Model loaded (ONNX): {model_id}")
            return True, f"Model '{spec.name}' loaded successfully via ONNX Runtime"
        except Exception as e:
            spec.status = ModelStatus.ERROR
            spec.load_error = str(e)
            log.error(f"Failed to load ONNX model {model_id}: {e}")
            return False, f"Failed to load model: {e}"

    def _load_pytorch_model(self, model_id: str, spec: ModelSpec) -> tuple[bool, str]:
        """For pytorch-format models, defer to the specific engine (e.g. VisionEngine)."""
        spec.status = ModelStatus.REGISTERED
        return False, f"PyTorch model '{model_id}' loading handled by specific engine (not preloaded)"

    def get_loaded_model(self, model_id: str) -> Optional[Any]:
        return self._loaded_models.get(model_id)

    def unload_model(self, model_id: str) -> bool:
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            spec = self._registry.get(model_id)
            if spec:
                spec.status = ModelStatus.DOWNLOADED
            log.info(f"Model unloaded: {model_id}")
            return True
        return False

    def print_model_report(self) -> None:
        models = self.list_models()
        print(f"\n{'─' * 50}")
        print(f"  SPECTRA — MODEL REGISTRY ({len(models)} models)")
        print(f"{'─' * 50}")
        for m in models:
            present = "✓" if self.is_model_file_present(m.model_id) else "✗"
            loaded = "LOADED" if m.model_id in self._loaded_models else m.status.value.upper()
            print(f"\n  [{present}] {m.name} (v{m.version})")
            print(f"      ID       : {m.model_id}")
            print(f"      Task     : {m.task.value}")
            print(f"      Format   : {m.format.value}")
            print(f"      Devices  : {', '.join(m.supported_devices)}")
            print(f"      Memory   : {m.memory_requirement_mb} MB")
            print(f"      Status   : {loaded}")
            print(f"      Local    : {'YES' if m.local_only else 'NO'}")
        print()


# Singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
