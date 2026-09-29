"""
SPECTRA - Vision Page
Image upload + object detection with routing transparency.
"""

from __future__ import annotations
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QPixmap
from utils.logger import get_logger

log = get_logger("UI")


class InferenceWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self._path = image_path

    def run(self):
        try:
            from ai.vision.engine import VisionEngine
            from config import settings

            engine = VisionEngine()
            model_path = settings.MODEL_DIRECTORY / "mobilenetv2-7.onnx"
            if not model_path.exists():
                model_path = settings.MODEL_DIRECTORY / "yolov8n.onnx"

            if not model_path.exists():
                self.error.emit(
                    f"Model file not found: {model_path}\n\n"
                    "Download the model:\n"
                    "  python scripts/download_mobilenet.py"
                )
                return

            ok, msg = engine.load(model_path)
            if not ok:
                self.error.emit(f"Model load failed: {msg}")
                return

            result = engine.infer(self._path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class VisionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path = None
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        title = QLabel("VISION ENGINE")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #818cf8; letter-spacing: 3px;")
        layout.addWidget(title)

        sub = QLabel("Object detection · Local AI inference · Hardware-aware routing")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color: #475569; margin-top: -8px;")
        layout.addWidget(sub)

        content = QHBoxLayout()
        content.setSpacing(16)

        # Left: image panel
        left = QVBoxLayout()
        self._img_label = QLabel()
        self._img_label.setMinimumSize(480, 360)
        self._img_label.setMaximumSize(640, 480)
        self._img_label.setStyleSheet("""
            background: #1a1a28;
            border: 2px dashed #1e1e30;
            border-radius: 8px;
            color: #334155;
            font-size: 14px;
        """)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setText("Drop an image here\nor click Upload")
        left.addWidget(self._img_label)

        btn_row = QHBoxLayout()
        self._upload_btn = QPushButton("\U0001f4c1  Upload Image")
        self._upload_btn.setFixedHeight(40)
        self._upload_btn.setStyleSheet("""
            QPushButton {
                background: #1e1e30; color: #818cf8;
                border: 1px solid #2d2d48; border-radius: 6px;
                font-size: 12px; font-weight: 600; padding: 0 16px;
            }
            QPushButton:hover { background: #252540; }
        """)
        self._upload_btn.clicked.connect(self._upload_image)

        self._run_btn = QPushButton("\u25b6  Run Detection")
        self._run_btn.setFixedHeight(40)
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("""
            QPushButton {
                background: #4f46e5; color: white; border: none;
                border-radius: 6px; font-size: 12px; font-weight: 600; padding: 0 16px;
            }
            QPushButton:hover { background: #6366f1; }
            QPushButton:disabled { background: #1e1e30; color: #334155; }
        """)
        self._run_btn.clicked.connect(self._run_inference)

        btn_row.addWidget(self._upload_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        left.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setRange(0, 0)
        self._progress.setStyleSheet("""
            QProgressBar { background: #1a1a28; border-radius: 4px; height: 6px; }
            QProgressBar::chunk { background: #6366f1; border-radius: 4px; }
        """)
        left.addWidget(self._progress)

        content.addLayout(left)

        # Right: results panel
        right = QVBoxLayout()
        results_title = QLabel("RESULTS")
        results_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        results_title.setStyleSheet("color: #475569; letter-spacing: 2px;")
        right.addWidget(results_title)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMinimumWidth(360)
        self._results_text.setStyleSheet("""
            QTextEdit {
                background: #1a1a28; color: #e2e8f0;
                border: 1px solid #1e1e30; border-radius: 8px;
                font-family: 'Segoe UI Mono'; font-size: 11px; padding: 12px;
            }
        """)
        self._results_text.setPlaceholderText("Inference results appear here...")
        right.addWidget(self._results_text)

        content.addLayout(right)
        layout.addLayout(content)
        layout.addStretch()

    def _upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self._image_path = path
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            self._img_label.setPixmap(pixmap)
            self._run_btn.setEnabled(True)
            self._results_text.clear()

    def _run_inference(self):
        if not self._image_path:
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)

        # Show routing plan
        try:
            from core.router.router import WorkloadRequest, get_router, LatencyRequirement
            from core.model_manager.manager import ModelTask, Modality
            router = get_router()
            req = WorkloadRequest(
                modality=Modality.VISION,
                task=ModelTask.IMAGE_CLASSIFICATION,
                input_type="image",
                latency_requirement=LatencyRequirement.REALTIME,
                offline_required=True,
            )
            plan = router.route(req)
            self._results_text.setText(f"ROUTING PLAN\n{plan.explain()}\n\nRunning inference...")
        except Exception as e:
            self._results_text.setText(f"Routing info unavailable: {e}\n\nRunning inference...")

        self._thread = QThread()
        self._worker = InferenceWorker(self._image_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_result(self, result):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)

        lines = []
        lines.append("AI EXECUTION DETAILS")
        lines.append("-" * 36)
        lines.append(f"Model    : {result.model_name}")
        lines.append(f"Task     : {result.task}")
        lines.append(f"Runtime  : {result.runtime}")
        lines.append(f"Device   : {result.execution_device}")
        lines.append(f"Provider : {result.provider}")
        lines.append(f"Latency  : {result.inference_time_ms:.2f} ms")
        lines.append(f"Local    : YES")
        lines.append(f"Image    : {result.image_width}x{result.image_height}")
        lines.append("")
        if result.task == "classification":
            lines.append(f"TOP PREDICTIONS ({len(result.classifications)})")
            lines.append("-" * 36)
            for c in result.classifications:
                lines.append(f"[{c.confidence:.2%}] {c.class_name} (id={c.class_id})")
        else:
            lines.append(f"DETECTIONS ({len(result.detections)})")
            lines.append("-" * 36)
            if result.detections:
                for d in result.detections:
                    lines.append(f"[{d.confidence:.1%}] {d.class_name}")
                    lines.append(f"       bbox: {[round(v, 0) for v in d.bbox_xyxy]}")
            else:
                lines.append("No objects detected above threshold")

        self._results_text.setText("\n".join(lines))

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._results_text.setText(f"ERROR\n{'─'*36}\n{msg}")
