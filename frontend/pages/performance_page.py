"""
SPECTRA - Performance Lab Page
Real benchmark measurements — no fabricated values.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSpinBox,
)
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui import QFont
from utils.logger import get_logger

log = get_logger("BENCHMARK")


class BenchmarkWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, model_id: str, warmup: int, iterations: int):
        super().__init__()
        self._model_id = model_id
        self._warmup = warmup
        self._iterations = iterations

    def run(self):
        import numpy as np
        from benchmarking.benchmark import run_benchmark
        from config import settings

        self.progress.emit("Checking model file...")
        model_path = settings.MODEL_DIRECTORY / "mobilenetv2-7.onnx"
        if not model_path.exists():
            model_path = settings.MODEL_DIRECTORY / "yolov8n.onnx"

        if not model_path.exists():
            self.error.emit(
                f"Model not found: {model_path}\n\n"
                "Download: python scripts/download_mobilenet.py"
            )
            return

        try:
            import onnxruntime as ort
            self.progress.emit("Loading ONNX session...")
            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            
            # Determine input size dynamically from model shape
            c = input_shape[1] if isinstance(input_shape[1], int) else 3
            h = input_shape[2] if isinstance(input_shape[2], int) else 224
            w = input_shape[3] if isinstance(input_shape[3], int) else 224
            dummy = np.random.rand(1, c, h, w).astype(np.float32)

            def inference_fn():
                sess.run(None, {input_name: dummy})

            model_name = "MobileNetV2 (ONNX)" if "mobilenet" in model_path.name.lower() else "YOLOv8 Nano"
            self.progress.emit(f"Running {self._warmup} warmup + {self._iterations} benchmark iterations on {model_name}...")
            result = run_benchmark(
                inference_fn=inference_fn,
                model_id=self._model_id,
                model_name=model_name,
                runtime="onnx",
                device="CPU",
                provider="CPUExecutionProvider",
                warmup_iterations=self._warmup,
                benchmark_iterations=self._iterations,
                notes="SPECTRA Performance Lab",
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class PerformancePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        title = QLabel("PERFORMANCE LAB")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #818cf8; letter-spacing: 3px;")
        layout.addWidget(title)

        sub = QLabel("Real benchmark measurements — no fabricated numbers")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color: #475569; margin-top: -8px;")
        layout.addWidget(sub)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("MobileNetV2 (mobilenet_v2)")
        self._model_combo.addItem("YOLOv8 Nano (yolov8n_onnx)")
        self._model_combo.setStyleSheet(
            "QComboBox { background: #1a1a28; color: #e2e8f0; border: 1px solid #1e1e30; "
            "border-radius: 4px; padding: 4px 8px; min-width: 200px; }"
        )
        controls.addWidget(self._model_combo)

        controls.addWidget(QLabel("Warmup:"))
        self._warmup_spin = QSpinBox()
        self._warmup_spin.setRange(1, 20)
        self._warmup_spin.setValue(3)
        self._warmup_spin.setStyleSheet(
            "QSpinBox { background: #1a1a28; color: #e2e8f0; border: 1px solid #1e1e30; "
            "border-radius: 4px; padding: 4px 8px; }"
        )
        controls.addWidget(self._warmup_spin)

        controls.addWidget(QLabel("Iterations:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 100)
        self._iter_spin.setValue(10)
        self._iter_spin.setStyleSheet(
            "QSpinBox { background: #1a1a28; color: #e2e8f0; border: 1px solid #1e1e30; "
            "border-radius: 4px; padding: 4px 8px; }"
        )
        controls.addWidget(self._iter_spin)

        self._run_btn = QPushButton("\u25b6  Run Benchmark")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setStyleSheet("""
            QPushButton {
                background: #4f46e5; color: white; border: none;
                border-radius: 6px; font-weight: 600; padding: 0 16px;
            }
            QPushButton:hover { background: #6366f1; }
            QPushButton:disabled { background: #1e1e30; color: #334155; }
        """)
        self._run_btn.clicked.connect(self._run_benchmark)
        controls.addWidget(self._run_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setStyleSheet("""
            QTextEdit {
                background: #1a1a28; color: #e2e8f0;
                border: 1px solid #1e1e30; border-radius: 8px;
                font-family: 'Segoe UI Mono'; font-size: 11px; padding: 16px;
            }
        """)
        self._results.setPlaceholderText(
            "Benchmark results appear here.\n\n"
            "Note: YOLOv8 model must be downloaded first.\n"
            "Run: python scripts/download_models.py"
        )
        layout.addWidget(self._results)

    def _run_benchmark(self):
        self._run_btn.setEnabled(False)
        self._results.setText("Starting benchmark...\n")

        self._thread = QThread()
        self._worker = BenchmarkWorker(
            "yolov8n_onnx",
            self._warmup_spin.value(),
            self._iter_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(lambda msg: self._results.append(msg))
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self._run_btn.setEnabled(True))
        self._thread.start()

    def _on_result(self, result):
        lines = []
        lines.append("=" * 50)
        lines.append("  SPECTRA PERFORMANCE LAB RESULTS")
        lines.append("=" * 50)
        lines.append(f"  Model     : {result.model_name}")
        lines.append(f"  Runtime   : {result.runtime}")
        lines.append(f"  Device    : {result.device}")
        lines.append(f"  Provider  : {result.provider}")
        lines.append(f"  Warmup    : {result.warmup_iterations} iterations")
        lines.append(f"  Benchmark : {result.benchmark_iterations} iterations")
        lines.append(f"  Timestamp : {result.timestamp}")
        lines.append("")
        lines.append("  LATENCY (real measurements)")
        lines.append(f"  Average  : {result.avg_latency_ms:.2f} ms")
        lines.append(f"  Min      : {result.min_latency_ms:.2f} ms")
        lines.append(f"  Max      : {result.max_latency_ms:.2f} ms")
        lines.append(f"  P50      : {result.p50_latency_ms:.2f} ms")
        lines.append(f"  P95      : {result.p95_latency_ms:.2f} ms")
        lines.append(f"  P99      : {result.p99_latency_ms:.2f} ms")
        lines.append(f"  Throughput: {result.throughput_rps:.2f} req/s")
        if result.memory_delta_mb is not None:
            lines.append(f"  Mem delta : +{result.memory_delta_mb:.1f} MB")
        lines.append("=" * 50)
        self._results.setText("\n".join(lines))

    def _on_error(self, msg: str):
        self._results.setText(f"BENCHMARK ERROR\n{'─'*40}\n{msg}")
        log.error(f"Benchmark error: {msg}")
