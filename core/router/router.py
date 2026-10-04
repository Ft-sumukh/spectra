"""
SPECTRA - Workload Router
MODULE D

The CORE FEATURE of SPECTRA.
Routes AI workloads to the optimal hardware/runtime combination.
Provides transparent reasoning for every routing decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from runtime.manager import ExecutionDevice, RuntimeType, get_runtime_manager
from core.model_manager.manager import ModelTask, Modality, ModelFormat, get_model_manager
from utils.logger import get_logger

log = get_logger("ROUTER")


class LatencyRequirement(str, Enum):
    REALTIME = "realtime"       # < 100ms
    INTERACTIVE = "interactive" # < 500ms
    BATCH = "batch"             # No strict limit


@dataclass
class WorkloadRequest:
    """Describes an AI workload to be routed."""
    modality: Modality
    task: ModelTask
    input_type: str
    latency_requirement: LatencyRequirement = LatencyRequirement.INTERACTIVE
    memory_requirement_mb: float = 0.0
    offline_required: bool = True
    preferred_device: Optional[ExecutionDevice] = None
    model_id_hint: Optional[str] = None  # Suggest a specific model

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "task": self.task.value,
            "input_type": self.input_type,
            "latency_requirement": self.latency_requirement.value,
            "memory_requirement_mb": self.memory_requirement_mb,
            "offline_required": self.offline_required,
            "preferred_device": self.preferred_device.value if self.preferred_device else None,
            "model_id_hint": self.model_id_hint,
        }


@dataclass
class ExecutionPlan:
    """
    The router's output — a complete execution plan with transparent reasoning.
    Never silent. Always explains why a device/runtime was chosen or rejected.
    """
    # Selected model
    selected_model_id: str
    selected_model_name: str

    # Selected execution path
    selected_runtime: RuntimeType
    selected_device: ExecutionDevice
    selected_provider: str  # ONNX EP or torch device string

    # Fallback
    fallback_device: Optional[ExecutionDevice]
    fallback_runtime: Optional[RuntimeType]
    fallback_used: bool = False

    # Reasoning (human-readable, shown in UI)
    routing_steps: list[str] = field(default_factory=list)
    routing_decision: str = ""
    fallback_reason: Optional[str] = None

    # Estimates
    estimated_latency_ms: Optional[float] = None
    estimated_memory_mb: Optional[float] = None

    # Flags
    offline_capable: bool = True
    npu_attempted: bool = False
    gpu_attempted: bool = False

    def explain(self) -> str:
        """Return a human-readable routing explanation."""
        lines = ["ROUTING DECISION"]
        lines.append("─" * 40)
        lines.append(f"Model    : {self.selected_model_name}")
        lines.append(f"Runtime  : {self.selected_runtime.value}")
        lines.append(f"Device   : {self.selected_device.value}")
        lines.append(f"Provider : {self.selected_provider}")
        if self.fallback_used:
            lines.append(f"\n⚠ FALLBACK ACTIVE")
            lines.append(f"  Reason: {self.fallback_reason}")
        lines.append("\nROUTING STEPS")
        for i, step in enumerate(self.routing_steps, 1):
            lines.append(f"  {i}. {step}")
        lines.append(f"\nDECISION: {self.routing_decision}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "selected_model_id": self.selected_model_id,
            "selected_model_name": self.selected_model_name,
            "selected_runtime": self.selected_runtime.value,
            "selected_device": self.selected_device.value,
            "selected_provider": self.selected_provider,
            "fallback_device": self.fallback_device.value if self.fallback_device else None,
            "fallback_used": self.fallback_used,
            "routing_steps": self.routing_steps,
            "routing_decision": self.routing_decision,
            "fallback_reason": self.fallback_reason,
            "estimated_latency_ms": self.estimated_latency_ms,
            "offline_capable": self.offline_capable,
            "npu_attempted": self.npu_attempted,
        }


class WorkloadRouter:
    """
    Hardware-Aware AI Workload Router.

    For every WorkloadRequest:
    1. Select the best model for the task
    2. Check if the preferred device is available
    3. Check if the runtime supports that device
    4. If not, find the best fallback
    5. Generate a transparent ExecutionPlan with reasoning

    Never silently falls back — always explains what happened and why.
    """

    # Device preference order: NPU > GPU > CPU
    _DEVICE_PRIORITY = [ExecutionDevice.NPU, ExecutionDevice.GPU, ExecutionDevice.CPU]

    # Runtime preference per device
    _RUNTIME_PREFERENCE: dict[ExecutionDevice, list[RuntimeType]] = {
        ExecutionDevice.NPU: [RuntimeType.ONNX_QNN, RuntimeType.QNN_NATIVE],
        ExecutionDevice.GPU: [RuntimeType.ONNX_GPU, RuntimeType.PYTORCH_CUDA],
        ExecutionDevice.CPU: [RuntimeType.ONNX_CPU, RuntimeType.PYTORCH_CPU],
    }

    # Provider strings for ONNX
    _PROVIDER_MAP: dict[RuntimeType, str] = {
        RuntimeType.ONNX_QNN: "QNNExecutionProvider",
        RuntimeType.ONNX_GPU: "DmlExecutionProvider",
        RuntimeType.ONNX_CPU: "CPUExecutionProvider",
        RuntimeType.PYTORCH_CPU: "cpu",
        RuntimeType.PYTORCH_CUDA: "cuda",
        RuntimeType.QNN_NATIVE: "QNN_NATIVE",
    }

    def __init__(self):
        self._rt_manager = get_runtime_manager()
        self._model_manager = get_model_manager()

    def route(self, request: WorkloadRequest) -> ExecutionPlan:
        """Route a workload request to the best available execution plan."""
        log.info(f"Routing: task={request.task.value}, modality={request.modality.value}, "
                 f"preferred_device={request.preferred_device}")

        steps: list[str] = []

        # Step 1: Select model
        model_spec, model_step = self._select_model(request)
        steps.append(model_step)

        if model_spec is None:
            log.error("No suitable model found for request")
            return self._error_plan(request, steps, "No model available for this task")

        # Step 2: Determine device preference order
        device_order = self._build_device_order(request)
        steps.append(f"Device evaluation order: {[d.value for d in device_order]}")

        # Step 3: Try devices in priority order
        for device in device_order:
            runtime_type, provider, reason = self._try_device(device, model_spec)
            is_preferred = (request.preferred_device == device) if request.preferred_device else False
            is_fallback = (device != device_order[0]) or (
                request.preferred_device is not None and device != request.preferred_device
            )

            if device == ExecutionDevice.NPU:
                request_copy = request
                steps.append(f"NPU check: {'available' if runtime_type else 'NOT AVAILABLE'} — {reason}")

            if device == ExecutionDevice.GPU:
                steps.append(f"GPU check: {'available' if runtime_type else 'NOT AVAILABLE'} — {reason}")

            if runtime_type is not None:
                # Success — build plan
                fallback_used = is_fallback or (
                    request.preferred_device is not None and device != request.preferred_device
                )
                fallback_reason = None
                if fallback_used and request.preferred_device and device != request.preferred_device:
                    fallback_reason = (
                        f"Requested device: {request.preferred_device.value} — "
                        f"Actual device: {device.value} — "
                        f"Reason: {request.preferred_device.value} runtime unavailable or model incompatible"
                    )
                elif fallback_used:
                    # Auto-routing landed below the first choice in the priority
                    # order. That is still a fallback and must be explained.
                    skipped = [d.value for d in device_order
                               if device_order.index(d) < device_order.index(device)]
                    fallback_reason = (
                        f"Auto-routing evaluated [{' > '.join(d.value for d in device_order)}] "
                        f"and selected {device.value}. "
                        f"Unavailable ahead of it: {', '.join(skipped) if skipped else 'none'}."
                    )

                decision = (
                    f"{device.value} selected because "
                    f"{reason} and model supports it."
                )
                if fallback_used:
                    decision = f"FALLBACK to {device.value}. " + (fallback_reason or "")

                steps.append(f"Selected: {device.value} via {runtime_type.value}")

                # Build fallback info
                fallback_device = None
                fallback_rt = None
                remaining_devices = [d for d in device_order if d != device]
                if remaining_devices:
                    fb_d = remaining_devices[0]
                    fb_rt, _, _ = self._try_device(fb_d, model_spec)
                    fallback_device = fb_d
                    fallback_rt = fb_rt

                plan = ExecutionPlan(
                    selected_model_id=model_spec.model_id,
                    selected_model_name=model_spec.name,
                    selected_runtime=runtime_type,
                    selected_device=device,
                    selected_provider=provider,
                    fallback_device=fallback_device,
                    fallback_runtime=fallback_rt,
                    fallback_used=fallback_used,
                    routing_steps=steps,
                    routing_decision=decision,
                    fallback_reason=fallback_reason,
                    estimated_memory_mb=model_spec.memory_requirement_mb,
                    offline_capable=model_spec.local_only,
                    npu_attempted=ExecutionDevice.NPU in device_order[:device_order.index(device)+1],
                    gpu_attempted=ExecutionDevice.GPU in device_order[:device_order.index(device)+1],
                )

                log.info(f"Routing complete: model={model_spec.model_id}, "
                         f"device={device.value}, runtime={runtime_type.value}")
                if fallback_used:
                    log.warning(f"Fallback used: {fallback_reason}")

                return plan

        # All devices failed
        steps.append("All devices exhausted — no compatible runtime found")
        return self._error_plan(request, steps, "No compatible runtime available for any device")

    def _select_model(self, request: WorkloadRequest) -> tuple:
        """Select the best model for the task."""
        # Use hint if provided
        if request.model_id_hint:
            spec = self._model_manager.get_model_spec(request.model_id_hint)
            if spec:
                return spec, f"Model '{spec.name}' selected (user hint: {request.model_id_hint})"

        # Find models for this task
        candidates = self._model_manager._registry.list_by_task(request.task)
        if not candidates:
            return None, f"No models registered for task '{request.task.value}'"

        # Filter by modality
        candidates = [m for m in candidates if m.modality == request.modality] or candidates

        # Prefer lower memory
        candidates.sort(key=lambda m: m.memory_requirement_mb)
        selected = candidates[0]
        return selected, f"Model '{selected.name}' selected for task '{request.task.value}'"

    def _build_device_order(self, request: WorkloadRequest) -> list[ExecutionDevice]:
        """Build device evaluation order based on preferences and flags."""
        from config import settings

        preferred = request.preferred_device
        order = []

        if preferred:
            order.append(preferred)
            for d in self._DEVICE_PRIORITY:
                if d != preferred:
                    order.append(d)
        else:
            # Auto-routing: NPU > GPU > CPU
            for d in self._DEVICE_PRIORITY:
                if d == ExecutionDevice.NPU and not settings.ENABLE_NPU:
                    continue
                if d == ExecutionDevice.GPU and not settings.ENABLE_GPU:
                    continue
                if d == ExecutionDevice.CPU and not settings.ENABLE_CPU:
                    continue
                order.append(d)

        return order

    def _try_device(self, device: ExecutionDevice, model_spec) -> tuple[Optional[RuntimeType], str, str]:
        """
        Try to find a working runtime for the given device and model.
        Returns (runtime_type, provider_string, reason) or (None, '', reason).
        """
        preferred_runtimes = self._RUNTIME_PREFERENCE.get(device, [])

        for rt_type in preferred_runtimes:
            rt_info = self._rt_manager.get_runtime(rt_type)
            if rt_info and rt_info.available:
                # Check model supports this runtime
                rt_name = rt_type.value
                if rt_name in model_spec.supported_runtimes or device.value in model_spec.supported_devices:
                    provider = self._PROVIDER_MAP.get(rt_type, "unknown")
                    return rt_type, provider, f"{rt_type.value} runtime available and model compatible"

        return None, "", f"No available runtime for {device.value} that supports this model"

    def _error_plan(self, request: WorkloadRequest, steps: list[str], reason: str) -> ExecutionPlan:
        """Return an error execution plan."""
        return ExecutionPlan(
            selected_model_id="NONE",
            selected_model_name="NO MODEL",
            selected_runtime=RuntimeType.UNKNOWN,
            selected_device=ExecutionDevice.UNKNOWN,
            selected_provider="NONE",
            fallback_device=None,
            fallback_runtime=None,
            fallback_used=False,
            routing_steps=steps,
            routing_decision=f"ROUTING FAILED: {reason}",
            fallback_reason=reason,
            offline_capable=False,
        )

    def explain_request(self, request: WorkloadRequest) -> str:
        """Route and explain without executing inference."""
        plan = self.route(request)
        return plan.explain()


# Singleton
_router: Optional[WorkloadRouter] = None


def get_router() -> WorkloadRouter:
    global _router
    if _router is None:
        _router = WorkloadRouter()
    return _router
