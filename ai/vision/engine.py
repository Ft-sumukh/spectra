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
from utils.coco_labels import COCO_CLASSES

# Upper bound for a plausible class count, used to identify which axis of a
# YOLO output tensor holds classes. Generous, but far below anchor counts
# (8400 for a 640x640 YOLOv8 head).
_MAX_CLASSES = 1000

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
    requested_device: Optional[str] = None
    fell_back_to_cpu: bool = False

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
            "requested_device": self.requested_device,
            "fell_back_to_cpu": self.fell_back_to_cpu,
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
        if self.requested_device and self.requested_device != self.execution_device:
            print(f"  Requested: {self.requested_device} (FALLBACK)")
        if self.fell_back_to_cpu and self.notes:
            print(f"  Note     : {self.notes}")
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
        auto_route: bool = True,
    ):
        self._model_id = model_id
        self._providers = providers
        self._conf_threshold = conf_threshold
        self._session = None
        self._input_name = None
        self._input_size = (224, 224)
        self._loaded = False
        self._device = "CPU"
        self._runtime = "onnx"
        self._provider_used = "CPUExecutionProvider"
        self._task = "classification"
        self._labels = ensure_imagenet_labels()
        # When True, load() asks the WorkloadRouter which device should run this
        # model and honours that decision. This is what makes routing real
        # rather than advisory.
        self._auto_route = auto_route
        self._route_plan = None
        self._requested_device = None
        self._fell_back = False
        self._fallback_reason = None

    @property
    def fell_back_to_cpu(self) -> bool:
        """True when a non-CPU device was requested but CPU is actually running."""
        return self._fell_back

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def provider(self) -> str:
        """The execution provider this model is actually bound to."""
        return self._provider_used

    @property
    def requested_device(self) -> Optional[str]:
        """The device the router asked for, regardless of what was bound."""
        return self._requested_device

    @property
    def route_plan(self):
        """The ExecutionPlan used to bind this engine, if routing was used."""
        return self._route_plan

    def routing_notes(self) -> str:
        """One-paragraph human explanation of how this model got bound."""
        lines = [
            f"Requested device : {self._requested_device or 'auto'}",
            f"Bound device     : {self._device}",
            f"Execution provider: {self._provider_used}",
        ]
        if self._fell_back:
            lines.append(f"FALLBACK         : {self._fallback_reason or 'NPU/GPU unavailable'}")
        if self._route_plan is not None:
            lines.append(f"Router decision  : {self._route_plan.routing_decision}")
        return "\n".join(lines)

    def load(self, model_path: Union[str, Path]) -> tuple[bool, str]:
        """Load an ONNX model for vision inference."""
        try:
            import onnxruntime as ort
            model_path = Path(model_path)
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"

            from core.execution import create_session

            requested = self._providers[0] if self._providers else None

            # Ask the router which silicon should run this task.
            if self._auto_route and requested is None:
                self._requested_device, plan = self._route_vision_task()
                self._route_plan = plan
            else:
                self._requested_device = "CPU"

            log.info(
                f"Loading vision model: {model_path.name} | "
                f"requested_device={self._requested_device}"
            )

            built = create_session(model_path, preferred_device=self._requested_device or "CPU")
            if not built.ok:
                return False, built.error or "Unknown session error"

            self._session = built.session
            self._provider_used = built.resolution.primary
            self._device = built.resolution.device
            self._fell_back = built.resolution.fell_back_to_cpu
            self._fallback_reason = "; ".join(built.resolution.notes) or None
            
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

            self._runtime = "onnx"
            self._loaded = True

            log.info(
                f"Vision model loaded ({self._task}). Provider: {self._provider_used}, "
                f"Device: {self._device}, fell_back={self._fell_back}"
            )
            msg = f"Model loaded: {model_path.name} ({self._task}) on {self._device} via {self._provider_used}"
            if self._fell_back:
                msg += f" [FALLBACK: {self._fallback_reason}]"
            return True, msg

        except ImportError:
            return False, "onnxruntime not installed."
        except Exception as e:
            log.error(f"Failed to load vision model: {e}")
            return False, f"Model load failed: {e}"

    def _route_vision_task(self) -> tuple[Optional[str], object]:
        """
        Ask the WorkloadRouter where this vision task should run.

        Returns (requested_device, execution_plan). Never raises: if routing
        itself fails we fall back to CPU and say so.
        """
        try:
            from core.router.router import (
                WorkloadRequest, LatencyRequirement, get_router,
            )
            from core.model_manager.manager import ModelTask, Modality

            task = (
                ModelTask.OBJECT_DETECTION if self._task == "object_detection"
                else ModelTask.IMAGE_CLASSIFICATION
            )
            request = WorkloadRequest(
                modality=Modality.VISION,
                task=task,
                input_type="image",
                latency_requirement=LatencyRequirement.REALTIME,
                offline_required=True,
                model_id_hint=self._model_id,
            )
            plan = get_router().route(request)
            device = plan.selected_device.value if plan.selected_device else "CPU"
            if device == "UNKNOWN":
                device = "CPU"
            return device, plan
        except Exception as e:
            log.warning(f"Routing failed ({e}); defaulting vision task to CPU")
            return "CPU", None

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

    # ── Object detection (YOLOv8) ───────────────────────────────────────────

    def _preprocess_detection(self, image) -> tuple[np.ndarray, float, float, int, int]:
        """
        Letterbox an image into the model's square input.

        YOLO exports expect [0, 1] scaling with no mean/std normalisation.
        Letterboxing preserves aspect ratio and keeps the box-to-image
        transform (scale + padding) needed to map predictions back.
        """
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
        tw, th = self._input_size
        scale = min(tw / orig_w, th / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        pad_x, pad_y = (tw - new_w) / 2, (th - new_h) / 2

        resized = pil_img.resize((new_w, new_h))
        canvas = PILImage.new("RGB", (tw, th), (114, 114, 114))
        canvas.paste(resized, (int(pad_x), int(pad_y)))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr.astype(np.float32), scale, (pad_x, pad_y), orig_w, orig_h

    def _postprocess_detection(
        self,
        output: np.ndarray,
        scale: float,
        padding: tuple,
        orig_w: int,
        orig_h: int,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_det: int = 50,
    ) -> list:
        """
        Decode YOLO output into labelled boxes.

        Two export styles are in circulation and both must work:

        * raw head    -- [1, 4+nc, anchors] (YOLOv8). NMS applied here.
        * end-to-end  -- [1, max_det, 6] as x1,y1,x2,y2,conf,class
          (YOLOv10). NMS already applied inside the graph.

        We branch on the trailing dimension: 6 columns means end-to-end,
        anything else is treated as a raw head.
        """
        out = np.asarray(output)

        if out.ndim == 3 and out.shape[-1] == 6 and out.shape[1] <= 1000:
            return self._decode_end_to_end(
                out, scale, padding, orig_w, orig_h, conf_threshold, max_det
            )
        return self._decode_raw_head(
            out, scale, padding, orig_w, orig_h,
            conf_threshold, iou_threshold, max_det
        )

    def _decode_end_to_end(
        self,
        out: np.ndarray,
        scale: float,
        padding: tuple,
        orig_w: int,
        orig_h: int,
        conf_threshold: float,
        max_det: int,
    ) -> list:
        """Decode an NMS'd [1, N, 6] export (YOLOv10-style)."""
        rows = np.asarray(out).reshape(-1, 6)
        pad_x, pad_y = padding
        results = []

        for x1, y1, x2, y2, conf, cid in rows:
            if float(conf) < conf_threshold:
                continue
            cid = int(cid)
            results.append(Detection(
                class_id=cid,
                class_name=(COCO_CLASSES[cid] if cid < len(COCO_CLASSES)
                            else f"Class {cid}"),
                confidence=float(conf),
                bbox_xyxy=[
                    float(np.clip((x1 - pad_x) / scale, 0, orig_w)),
                    float(np.clip((y1 - pad_y) / scale, 0, orig_h)),
                    float(np.clip((x2 - pad_x) / scale, 0, orig_w)),
                    float(np.clip((y2 - pad_y) / scale, 0, orig_h)),
                ],
            ))

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results[:max_det]

    def _decode_raw_head(
        self,
        out: np.ndarray,
        scale: float,
        padding: tuple,
        orig_w: int,
        orig_h: int,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
    ) -> list:
        """Decode a raw [1, 4+nc, anchors] head and run NMS (YOLOv8-style)."""
        out = np.asarray(out)
        while out.ndim > 2:
            out = np.squeeze(out, axis=0)
        if out.ndim != 2:
            raise ValueError(
                f"Unexpected YOLO output shape: {np.asarray(out).shape}"
            )

        # Exports come as either [anchors, 4+nc] or [4+nc, anchors]. Rather
        # than guess from which side is bigger, ask which dimension minus 4
        # looks like a class count. Comparing sizes alone fails on
        # [84, 4]-style tensors, where the class dimension is the *first* one.
        if not (1 <= out.shape[1] - 4 <= _MAX_CLASSES):
            if 1 <= out.shape[0] - 4 <= _MAX_CLASSES:
                out = out.T

        nc = out.shape[1] - 4
        if nc <= 0:
            raise ValueError(f"YOLO output has no class dimension: {out.shape}")

        scores = out[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        keep = confidences >= conf_threshold
        if not np.any(keep):
            return []
        boxes_xywh = out[keep, :4]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # cx, cy, w, h live in letterboxed pixels -> map back to the original.
        pad_x, pad_y = padding
        cx = (boxes_xywh[:, 0] - pad_x) / scale
        cy = (boxes_xywh[:, 1] - pad_y) / scale
        bw = boxes_xywh[:, 2] / scale
        bh = boxes_xywh[:, 3] / scale

        x1 = np.clip(cx - bw / 2, 0, orig_w)
        y1 = np.clip(cy - bh / 2, 0, orig_h)
        x2 = np.clip(cx + bw / 2, 0, orig_w)
        y2 = np.clip(cy + bh / 2, 0, orig_h)

        detections = []
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            boxes = np.stack([x1[idx], y1[idx], x2[idx], y2[idx]], axis=1)
            for i in _nms(boxes, confidences[idx], iou_threshold):
                detections.append(Detection(
                    class_id=int(cid),
                    class_name=(COCO_CLASSES[cid] if cid < len(COCO_CLASSES)
                                else f"Class {cid}"),
                    confidence=float(confidences[idx][i]),
                    bbox_xyxy=[float(v) for v in boxes[i]],
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:max_det]

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
            if self._task == "object_detection":
                return self._infer_detection(image)

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
                requested_device=self._requested_device,
                fell_back_to_cpu=self._fell_back,
                notes=self._fallback_reason or "",
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

    def _infer_detection(self, image) -> VisionResult:
        """Run object detection and return real boxes."""
        try:
            img_tensor, scale, padding, orig_w, orig_h = self._preprocess_detection(image)

            t0 = time.perf_counter()
            outputs = self._session.run(None, {self._input_name: img_tensor})
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            detections = self._postprocess_detection(
                outputs[0], scale, padding, orig_w, orig_h,
                conf_threshold=self._conf_threshold if self._conf_threshold > 0.1 else 0.35,
            )

            log.info(
                f"Detection: {latency_ms:.2f}ms | {len(detections)} objects | "
                f"device={self._device}"
            )

            return VisionResult(
                success=True,
                task="object_detection",
                detections=detections,
                inference_time_ms=latency_ms,
                execution_device=self._device,
                runtime=self._runtime,
                provider=self._provider_used,
                model_name=getattr(self, "_model_name", "YOLOv8 Nano"),
                model_id=self._model_id,
                image_width=orig_w,
                image_height=orig_h,
                requested_device=self._requested_device,
                fell_back_to_cpu=self._fell_back,
                notes=self._fallback_reason or "",
            )
        except Exception as e:
            log.error(f"Detection error: {e}")
            return VisionResult(
                success=False,
                task="object_detection",
                inference_time_ms=0.0,
                execution_device=self._device,
                runtime=self._runtime,
                provider=self._provider_used,
                model_name=getattr(self, "_model_name", "YOLOv8 Nano"),
                model_id=self._model_id,
                error_message=str(e),
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Greedy non-maximum suppression.

    Returns indices into `boxes`, highest score first.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_threshold]
    return keep
