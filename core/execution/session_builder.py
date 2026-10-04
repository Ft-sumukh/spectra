"""
SPECTRA - ONNX Session Builder

Builds real onnxruntime.InferenceSession objects from a router decision.

Key guarantees:
1. The provider list we request is filtered against providers ONNX Runtime
   actually compiled in, so we never request an EP that does not exist.
2. We always append CPUExecutionProvider last as a genuine safety net, so a
   model that the NPU cannot fully offload still produces a result instead of
   raising. When that happens we report it honestly via `fell_back_to_cpu`.
3. After session creation we read back `session.get_providers()` and record
   what was ACTUALLY bound, rather than assuming the request was honoured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("EXEC")

# Always the last resort. Never removed from a provider list.
CPU_EP = "CPUExecutionProvider"

# Provider -> device. Used to translate what ORT reports back into a device.
_PROVIDER_DEVICE = {
    "QNNExecutionProvider": "NPU",
    "NvTensorRtRtxExecutionProvider": "GPU",
    "CUDAExecutionProvider": "GPU",
    "DmlExecutionProvider": "GPU",
    "ROCMExecutionProvider": "GPU",
    "OpenVINOExecutionProvider": "GPU",
    "CPUExecutionProvider": "CPU",
}

# Devices we are willing to request, in the order we trust them.
_EP_PRIORITY = {
    "NPU": ["QNNExecutionProvider"],
    "GPU": ["CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider"],
    "CPU": ["CPUExecutionProvider"],
}


def provider_device_map(provider: str) -> str:
    """Map an execution provider name to the device it runs on."""
    return _PROVIDER_DEVICE.get(provider, "UNKNOWN")


@dataclass
class ProviderResolution:
    """What we asked for, what we got, and what the difference means."""

    requested: list[str] = field(default_factory=list)
    granted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # not compiled into this ORT
    primary: str = CPU_EP
    device: str = "CPU"
    fell_back_to_cpu: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "granted": self.granted,
            "rejected": self.rejected,
            "primary": self.primary,
            "device": self.device,
            "fell_back_to_cpu": self.fell_back_to_cpu,
            "notes": self.notes,
        }

    def explain(self) -> str:
        lines = [
            "PROVIDER RESOLUTION",
            "-" * 40,
            f"Requested : {self.requested}",
            f"Granted   : {self.granted}",
        ]
        if self.rejected:
            lines.append(f"Rejected  : {self.rejected} (not compiled into this ONNX Runtime)")
        lines.append(f"Primary   : {self.primary}")
        lines.append(f"Device    : {self.device}")
        if self.fell_back_to_cpu:
            lines.append("FALLBACK  : Requested EP was unavailable; execution bound to CPU.")
        for n in self.notes:
            lines.append(f"Note      : {n}")
        return "\n".join(lines)


@dataclass
class ExecutionSession:
    """A live ONNX session plus an honest record of where it is running."""

    session: Any = None
    resolution: ProviderResolution = field(default_factory=ProviderResolution)
    model_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.session is not None and self.error is None

    @property
    def device(self) -> str:
        return self.resolution.device

    @property
    def provider(self) -> str:
        return self.resolution.primary

    def run(self, feed: dict) -> list:
        if not self.ok:
            raise RuntimeError(self.error or "Session not initialised")
        return self.session.run(None, feed)


def build_providers(
    preferred_device: str = "CPU",
    available_providers: Optional[list[str]] = None,
    provider_options: Optional[dict] = None,
) -> ProviderResolution:
    """
    Build a provider list for the requested device.

    `available_providers` should come from `ort.get_available_providers()`.
    If omitted we query ONNX Runtime directly.
    """
    if available_providers is None:
        try:
            import onnxruntime as ort
            available_providers = list(ort.get_available_providers())
        except ImportError:
            available_providers = [CPU_EP]

    res = ProviderResolution()

    candidates = list(_EP_PRIORITY.get(preferred_device.upper(), []))
    if not candidates:
        res.notes.append(f"Unknown device '{preferred_device}'; defaulting to CPU.")
        candidates = [CPU_EP]

    requested: list[str] = []
    for ep in candidates:
        if ep in available_providers:
            requested.append(ep)
        else:
            res.rejected.append(ep)

    wanted_npu = preferred_device.upper() == "NPU"

    if not requested or requested == [CPU_EP]:
        if wanted_npu:
            res.fell_back_to_cpu = True
            res.notes.append(
                "NPU requested but no NPU execution provider is present in this "
                "ONNX Runtime build. On Snapdragon this means the Qualcomm AI Stack "
                "(onnxruntime-qnn) is not installed."
            )
        requested = [CPU_EP]

    # CPU always last so partial offload still runs.
    if CPU_EP not in requested:
        requested.append(CPU_EP)

    res.requested = requested
    if provider_options:
        res.notes.append(f"Provider options: {provider_options}")
    return res


def create_session(
    model_path: str | Path,
    preferred_device: str = "CPU",
    provider_options: Optional[dict] = None,
    session_options: Optional[dict] = None,
) -> ExecutionSession:
    """
    Create a real ONNX InferenceSession bound to the requested device.

    Never raises: on failure returns an ExecutionSession with `error` set.
    """
    path = Path(model_path)
    out = ExecutionSession(model_path=path)

    if not path.exists():
        out.error = f"Model file not found: {path}"
        log.error(out.error)
        return out

    try:
        import onnxruntime as ort
    except ImportError:
        out.error = "onnxruntime is not installed."
        log.error(out.error)
        return out

    res = build_providers(preferred_device, provider_options=provider_options)

    so = None
    if session_options:
        try:
            so = ort.SessionOptions()
            for k, v in session_options.items():
                setattr(so, k, v)
        except Exception as e:  # non-fatal
            log.warning(f"Could not apply session options: {e}")

    try:
        sess = ort.InferenceSession(str(path), sess_options=so, providers=res.requested)
    except Exception as e:
        log.error(f"Session creation failed with {res.requested}: {e}")
        out.error = f"Failed to create ONNX session: {e}"
        return out

    # Read back what ORT actually bound — do not assume the request was honoured.
    actual = list(sess.get_providers())
    res.granted = actual
    primary = actual[0] if actual else CPU_EP
    res.primary = primary
    res.device = provider_device_map(primary)

    if res.device == "CPU" and preferred_device.upper() in ("NPU", "GPU"):
        res.fell_back_to_cpu = True
        res.notes.append(
            f"Requested {preferred_device.upper()}, but ONNX Runtime bound the session "
            f"to {primary}. The model will run on CPU."
        )
    elif res.requested != actual:
        res.notes.append(f"Requested {res.requested} but ORT bound {actual}.")

    out.session = sess
    out.resolution = res
    log.info(
        f"Session ready: {path.name} | device={res.device} | primary={primary} | "
        f"granted={actual}"
    )
    return out
