"""
SPECTRA - Dashboard Page
Shows hardware, runtime, model, and execution status at a glance.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QScrollArea,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
    lbl.setStyleSheet("color: #475569; letter-spacing: 2px; padding-bottom: 6px;")
    return lbl


def _make_card(title: str, value: str, color: str = "#e2e8f0", subtitle: str = "") -> QFrame:
    frame = QFrame()
    frame.setStyleSheet("""
        QFrame {
            background: #1a1a28;
            border: 1px solid #1e1e30;
            border-radius: 8px;
        }
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(4)

    t = QLabel(title)
    t.setFont(QFont("Segoe UI", 8))
    t.setStyleSheet("color: #475569;")
    layout.addWidget(t)

    v = QLabel(value)
    v.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
    v.setStyleSheet(f"color: {color};")
    v.setWordWrap(True)
    layout.addWidget(v)

    if subtitle:
        s = QLabel(subtitle)
        s.setFont(QFont("Segoe UI", 8))
        s.setStyleSheet("color: #334155;")
        layout.addWidget(s)

    return frame


def _update_card(frame: QFrame, title: str, value: str, color: str, subtitle: str = ""):
    layout = frame.layout()
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    t = QLabel(title)
    t.setFont(QFont("Segoe UI", 8))
    t.setStyleSheet("color: #475569;")
    layout.addWidget(t)

    v = QLabel(value)
    v.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
    v.setStyleSheet(f"color: {color};")
    v.setWordWrap(True)
    layout.addWidget(v)

    if subtitle:
        s = QLabel(subtitle)
        s.setFont(QFont("Segoe UI", 8))
        s.setStyleSheet("color: #334155;")
        layout.addWidget(s)


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        QTimer.singleShot(300, self._refresh)

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #0f0f1a; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(24)

        # Header
        header = QLabel("SPECTRA")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #818cf8; letter-spacing: 6px;")
        layout.addWidget(header)

        sub = QLabel("Snapdragon-Powered Private Multimodal AI Workspace")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet("color: #475569; margin-top: -10px;")
        layout.addWidget(sub)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: #1e1e2e; margin: 4px 0;")
        layout.addWidget(div)

        # Hardware section
        layout.addWidget(_section_title("HARDWARE"))
        hw_grid = QGridLayout()
        hw_grid.setSpacing(12)

        self._cpu_card = _make_card("CPU", "Detecting...", "#94a3b8")
        self._gpu_card = _make_card("GPU", "Detecting...", "#94a3b8")
        self._npu_card = _make_card("NPU", "Detecting...", "#94a3b8")
        self._ram_card = _make_card("RAM", "Detecting...", "#94a3b8")

        hw_grid.addWidget(self._cpu_card, 0, 0)
        hw_grid.addWidget(self._gpu_card, 0, 1)
        hw_grid.addWidget(self._npu_card, 0, 2)
        hw_grid.addWidget(self._ram_card, 0, 3)
        layout.addLayout(hw_grid)

        # Runtime section
        layout.addWidget(_section_title("RUNTIME & ACCELERATION"))
        rt_grid = QGridLayout()
        rt_grid.setSpacing(12)
        self._ort_card = _make_card("ONNX Runtime", "Detecting...", "#94a3b8")
        self._qnn_card = _make_card("QNN / NPU Runtime", "Detecting...", "#94a3b8")
        self._torch_card = _make_card("PyTorch", "Detecting...", "#94a3b8")
        self._providers_card = _make_card("Execution Providers", "Detecting...", "#94a3b8")
        rt_grid.addWidget(self._ort_card, 0, 0)
        rt_grid.addWidget(self._qnn_card, 0, 1)
        rt_grid.addWidget(self._torch_card, 0, 2)
        rt_grid.addWidget(self._providers_card, 0, 3)
        layout.addLayout(rt_grid)

        # Models section
        layout.addWidget(_section_title("AI MODELS"))
        self._models_frame = QFrame()
        self._models_frame.setStyleSheet(
            "QFrame { background: #1a1a28; border: 1px solid #1e1e30; border-radius: 8px; }"
        )
        self._models_layout = QVBoxLayout(self._models_frame)
        self._models_layout.setContentsMargins(16, 12, 16, 12)
        self._models_layout.addWidget(QLabel("Loading..."))
        layout.addWidget(self._models_frame)

        # System section
        layout.addWidget(_section_title("SYSTEM"))
        sys_grid = QGridLayout()
        sys_grid.setSpacing(12)
        self._os_card = _make_card("Operating System", "Detecting...", "#94a3b8")
        self._arch_card = _make_card("Architecture", "Detecting...", "#94a3b8")
        self._py_card = _make_card("Python", "Detecting...", "#94a3b8")
        self._offline_card = _make_card("Mode", "Detecting...", "#94a3b8")
        sys_grid.addWidget(self._os_card, 0, 0)
        sys_grid.addWidget(self._arch_card, 0, 1)
        sys_grid.addWidget(self._py_card, 0, 2)
        sys_grid.addWidget(self._offline_card, 0, 3)
        layout.addLayout(sys_grid)
        layout.addStretch()

    def _refresh(self):
        # Hardware
        try:
            from hardware.detector import detect_hardware
            hw = detect_hardware()

            cpu_sub = f"{hw.cpu.physical_cores}P / {hw.cpu.logical_cores}L cores"
            _update_card(self._cpu_card, "CPU", hw.cpu.name[:40], "#e2e8f0", cpu_sub)

            if hw.gpus:
                g = hw.gpus[0]
                gpu_sub = f"{g.vram_mb:.0f} MB VRAM" if g.vram_mb else "VRAM: unknown"
                _update_card(self._gpu_card, "GPU", g.name[:40], "#e2e8f0", gpu_sub)
            else:
                _update_card(self._gpu_card, "GPU", "Not detected", "#64748b")

            if hw.npu.available:
                _update_card(self._npu_card, "NPU", hw.npu.name or "AVAILABLE",
                             "#4ade80", hw.npu.framework or "")
            else:
                _update_card(self._npu_card, "NPU", "NOT DETECTED", "#ef4444",
                             "Requires Snapdragon + QNN")

            ram_sub = f"{hw.ram.available_mb:.0f} MB available"
            _update_card(self._ram_card, "RAM",
                         f"{hw.ram.total_mb/1024:.1f} GB", "#e2e8f0", ram_sub)

            _update_card(self._os_card, "Operating System",
                         hw.os_name, "#e2e8f0", hw.os_version[:40])
            _update_card(self._arch_card, "Architecture",
                         hw.cpu.architecture, "#e2e8f0", hw.cpu.vendor)
            _update_card(self._py_card, "Python",
                         hw.python_version.split()[0], "#e2e8f0")
        except Exception:
            pass

        # Runtime
        try:
            from runtime.manager import get_runtime_manager, RuntimeType
            rtm = get_runtime_manager()
            providers = rtm.get_ort_providers()

            ort_rt = rtm.get_runtime(RuntimeType.ONNX_CPU)
            ort_color = "#4ade80" if (ort_rt and ort_rt.available) else "#ef4444"
            _update_card(self._ort_card, "ONNX Runtime",
                         ort_rt.version if ort_rt else "Not found", ort_color)

            qnn_rt = rtm.get_runtime(RuntimeType.ONNX_QNN)
            qnn_available = qnn_rt and qnn_rt.available
            qnn_color = "#4ade80" if qnn_available else "#ef4444"
            qnn_val = "AVAILABLE" if qnn_available else "NOT AVAILABLE"
            qnn_sub = "" if qnn_available else "Requires Snapdragon hardware"
            _update_card(self._qnn_card, "QNN / NPU Runtime", qnn_val, qnn_color, qnn_sub)

            torch_rt = rtm.get_runtime(RuntimeType.PYTORCH_CPU)
            torch_color = "#4ade80" if (torch_rt and torch_rt.available) else "#ef4444"
            _update_card(self._torch_card, "PyTorch",
                         torch_rt.version if torch_rt else "Not found", torch_color)

            providers_str = "\n".join(providers) if providers else "None"
            _update_card(self._providers_card, "Execution Providers",
                         providers_str, "#94a3b8")
        except Exception:
            pass

        # Models
        try:
            from core.model_manager.manager import get_model_manager
            mm = get_model_manager()
            models = mm.list_models()
            while self._models_layout.count():
                item = self._models_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            for m in models:
                present = "✓" if mm.is_model_file_present(m.model_id) else "✗ (not downloaded)"
                row = QLabel(f"{present}  {m.name}  [{m.task.value}]  {m.format.value.upper()}  {m.memory_requirement_mb:.0f}MB")
                row.setFont(QFont("Segoe UI Mono", 9))
                color = "#4ade80" if mm.is_model_file_present(m.model_id) else "#64748b"
                row.setStyleSheet(f"color: {color}; padding: 2px 0;")
                self._models_layout.addWidget(row)
        except Exception:
            pass

        # Offline mode
        try:
            from config import settings
            if settings.OFFLINE_MODE:
                _update_card(self._offline_card, "Mode", "OFFLINE", "#f59e0b", "Local inference only")
            else:
                _update_card(self._offline_card, "Mode", "ONLINE", "#4ade80", "Cloud APIs enabled")
        except Exception:
            pass
