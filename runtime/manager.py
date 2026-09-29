"""
SPECTRA - Runtime Manager
MODULE B

Detects available inference runtimes and execution providers.
Provides a clean abstraction for CPU / GPU / NPU runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from utils.logger import get_logger

log = get_logger("RUNTIME")


class RuntimeType(str, Enum):
    ONNX_CPU = "onnx_cpu"
    ONNX_GPU = "onnx_gpu"
    ONNX_QNN = "onnx_qnn"
    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_CUDA = "pytorch_cuda"
    OPENVINO = "openvino"
    QNN_NATIVE = "qnn_native"
    UNKNOWN = "unknown"


class ExecutionDevice(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    UNKNOWN = "UNKNOWN"


@dataclass
class RuntimeInfo:
    runtime_type: RuntimeType
    name: str
    version: Optional[str]
    available: bool
    device: ExecutionDevice
    providers: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "runtime_type": self.runtime_type.value,
            "name": self.name,
            "version": self.version,
            "available": self.available,
            "device": self.device.value,
            "providers": self.providers,
            "notes": self.notes,
        }


@dataclass
class ExecutionProviderInfo:
    name: str
    available: bool
    device: ExecutionDevice
    notes: str = ""


class RuntimeManager:
    """
    Detects and manages available inference runtimes.

    Runtimes are discovered at startup and cached. The manager provides
    the best available runtime for a given device preference.
    """

    def __init__(self):
        self._runtimes: dict[RuntimeType, RuntimeInfo] = {}
        self._ort_providers: list[str] = []
        self._detected = False

    def detect(self) -> None:
        """Run full runtime detection."""
        log.info("Starting runtime detection...")
        self._detect_onnx_runtime()
        self._detect_pytorch()
        self._detect_qnn()
        self._detect_openvino()
        self._detected = True
        self._log_summary()

    def _detect_onnx_runtime(self) -> None:
        try:
            import onnxruntime as ort
            version = ort.__version__
            providers = ort.get_available_providers()
            self._ort_providers = providers
            log.info(f"ONNX Runtime {version} found. Providers: {providers}")

            # CPU
            self._runtimes[RuntimeType.ONNX_CPU] = RuntimeInfo(
                runtime_type=RuntimeType.ONNX_CPU,
                name="ONNX Runtime (CPU)",
                version=version,
                available="CPUExecutionProvider" in providers,
                device=ExecutionDevice.CPU,
                providers=[p for p in providers if "CPU" in p],
                notes="Standard CPU inference"
            )

            # GPU (CUDA / DirectML)
            gpu_providers = [p for p in providers if p in
                            ("CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider")]
            if gpu_providers:
                self._runtimes[RuntimeType.ONNX_GPU] = RuntimeInfo(
                    runtime_type=RuntimeType.ONNX_GPU,
                    name=f"ONNX Runtime (GPU) via {gpu_providers[0]}",
                    version=version,
                    available=True,
                    device=ExecutionDevice.GPU,
                    providers=gpu_providers,
                    notes=f"GPU acceleration via {gpu_providers}"
                )
                log.info(f"GPU ONNX provider(s) available: {gpu_providers}")
            else:
                self._runtimes[RuntimeType.ONNX_GPU] = RuntimeInfo(
                    runtime_type=RuntimeType.ONNX_GPU,
                    name="ONNX Runtime (GPU)",
                    version=version,
                    available=False,
                    device=ExecutionDevice.GPU,
                    providers=[],
                    notes="No GPU execution provider (CUDA/DML/ROCm) found"
                )

            # QNN / NPU
            if "QNNExecutionProvider" in providers:
                self._runtimes[RuntimeType.ONNX_QNN] = RuntimeInfo(
                    runtime_type=RuntimeType.ONNX_QNN,
                    name="ONNX Runtime (QNN/NPU)",
                    version=version,
                    available=True,
                    device=ExecutionDevice.NPU,
                    providers=["QNNExecutionProvider"],
                    notes="Qualcomm Neural Network acceleration via ONNX Runtime"
                )
                log.info("QNNExecutionProvider available — Snapdragon NPU inference ready")
            else:
                self._runtimes[RuntimeType.ONNX_QNN] = RuntimeInfo(
                    runtime_type=RuntimeType.ONNX_QNN,
                    name="ONNX Runtime (QNN/NPU)",
                    version=version,
                    available=False,
                    device=ExecutionDevice.NPU,
                    providers=[],
                    notes="QNNExecutionProvider NOT available. Requires Qualcomm hardware + QNN SDK."
                )

        except ImportError:
            log.warning("ONNX Runtime not installed")
            for rt in (RuntimeType.ONNX_CPU, RuntimeType.ONNX_GPU, RuntimeType.ONNX_QNN):
                self._runtimes[rt] = RuntimeInfo(
                    runtime_type=rt, name=f"ONNX Runtime ({rt.value})",
                    version=None, available=False, device=ExecutionDevice.UNKNOWN,
                    notes="onnxruntime package not installed"
                )

    def _detect_pytorch(self) -> None:
        try:
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()

            self._runtimes[RuntimeType.PYTORCH_CPU] = RuntimeInfo(
                runtime_type=RuntimeType.PYTORCH_CPU,
                name="PyTorch (CPU)",
                version=version,
                available=True,
                device=ExecutionDevice.CPU,
                notes=f"PyTorch {version} CPU inference available"
            )
            log.info(f"PyTorch {version} detected. CUDA: {cuda_available}")

            self._runtimes[RuntimeType.PYTORCH_CUDA] = RuntimeInfo(
                runtime_type=RuntimeType.PYTORCH_CUDA,
                name="PyTorch (CUDA/GPU)",
                version=version,
                available=cuda_available,
                device=ExecutionDevice.GPU,
                notes=f"CUDA available: {cuda_available}. Device: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}"
            )

        except ImportError:
            log.warning("PyTorch not installed")
            self._runtimes[RuntimeType.PYTORCH_CPU] = RuntimeInfo(
                runtime_type=RuntimeType.PYTORCH_CPU, name="PyTorch (CPU)",
                version=None, available=False, device=ExecutionDevice.CPU,
                notes="torch package not installed"
            )

    def _detect_qnn(self) -> None:
        """Detect native Qualcomm QNN SDK (separate from ONNX Runtime QNN EP)."""
        try:
            import qnn_wrapper  # type: ignore
            self._runtimes[RuntimeType.QNN_NATIVE] = RuntimeInfo(
                runtime_type=RuntimeType.QNN_NATIVE,
                name="Qualcomm QNN SDK (Native)",
                version=getattr(qnn_wrapper, "__version__", "unknown"),
                available=True,
                device=ExecutionDevice.NPU,
                notes="Native QNN SDK available for direct Hexagon NPU access"
            )
            log.info("Native QNN SDK detected")
        except ImportError:
            self._runtimes[RuntimeType.QNN_NATIVE] = RuntimeInfo(
                runtime_type=RuntimeType.QNN_NATIVE,
                name="Qualcomm QNN SDK (Native)",
                version=None,
                available=False,
                device=ExecutionDevice.NPU,
                notes="qnn_wrapper not found. Install Qualcomm AI Stack on Snapdragon hardware."
            )

    def _detect_openvino(self) -> None:
        try:
            import openvino as ov
            version = ov.__version__
            self._runtimes[RuntimeType.OPENVINO] = RuntimeInfo(
                runtime_type=RuntimeType.OPENVINO,
                name="Intel OpenVINO",
                version=version,
                available=True,
                device=ExecutionDevice.CPU,  # Also GPU via iGPU
                notes=f"OpenVINO {version} — Intel CPU/iGPU acceleration"
            )
            log.info(f"OpenVINO {version} detected")
        except ImportError:
            self._runtimes[RuntimeType.OPENVINO] = RuntimeInfo(
                runtime_type=RuntimeType.OPENVINO,
                name="Intel OpenVINO",
                version=None,
                available=False,
                device=ExecutionDevice.CPU,
                notes="openvino package not installed"
            )

    def _log_summary(self) -> None:
        available = [r.name for r in self._runtimes.values() if r.available]
        unavailable = [r.name for r in self._runtimes.values() if not r.available]
        log.info(f"Available runtimes: {available}")
        if unavailable:
            log.info(f"Unavailable runtimes: {unavailable}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_runtime(self, rt_type: RuntimeType) -> Optional[RuntimeInfo]:
        if not self._detected:
            self.detect()
        return self._runtimes.get(rt_type)

    def list_runtimes(self) -> list[RuntimeInfo]:
        if not self._detected:
            self.detect()
        return list(self._runtimes.values())

    def list_available_runtimes(self) -> list[RuntimeInfo]:
        return [r for r in self.list_runtimes() if r.available]

    def get_ort_providers(self) -> list[str]:
        return self._ort_providers

    def best_runtime_for_device(self, device: ExecutionDevice) -> Optional[RuntimeInfo]:
        """Return the best available runtime for the given device preference."""
        candidates = [r for r in self.list_available_runtimes() if r.device == device]
        return candidates[0] if candidates else None

    def npu_available(self) -> bool:
        npu_runtimes = [r for r in self.list_available_runtimes()
                        if r.device == ExecutionDevice.NPU]
        return len(npu_runtimes) > 0

    def print_runtime_report(self) -> None:
        divider = "-" * 50
        print(f"\n{divider}")
        print("  SPECTRA - RUNTIME DETECTION REPORT")
        print(f"{divider}")
        for rt in self.list_runtimes():
            status = "[AVAILABLE]" if rt.available else "[UNAVAILABLE]"
            print(f"\n  [{rt.device.value}] {rt.name}")
            print(f"  Status  : {status}")
            print(f"  Version : {rt.version or 'N/A'}")
            if rt.providers:
                print(f"  EP(s)   : {', '.join(rt.providers)}")
            print(f"  Notes   : {rt.notes}")
        print(f"\n{divider}")
        print("  ONNX Runtime Execution Providers")
        print(f"{divider}")
        for p in self._ort_providers:
            print(f"  * {p}")
        print()


# Singleton
_runtime_manager: Optional[RuntimeManager] = None


def get_runtime_manager() -> RuntimeManager:
    global _runtime_manager
    if _runtime_manager is None:
        _runtime_manager = RuntimeManager()
        _runtime_manager.detect()
    return _runtime_manager
