"""
SPECTRA - Models Page
Displays all registered models with status and metadata.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QScrollArea, QHBoxLayout,
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont


class ModelCard(QFrame):
    def __init__(self, model_spec, is_present: bool, parent=None):
        super().__init__(parent)
        m = model_spec
        self.setStyleSheet(
            "QFrame { background: #1a1a28; border: 1px solid #1e1e30; border-radius: 8px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        header = QHBoxLayout()
        name = QLabel(m.name)
        name.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        name.setStyleSheet("color: #e2e8f0;")
        header.addWidget(name)

        status_color = "#4ade80" if is_present else "#f59e0b"
        status_text = "● READY" if is_present else "● NOT DOWNLOADED"
        status = QLabel(status_text)
        status.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        status.setStyleSheet(f"color: {status_color};")
        header.addStretch()
        header.addWidget(status)
        layout.addLayout(header)

        details = [
            ("ID",        m.model_id),
            ("Task",      m.task.value),
            ("Modality",  m.modality.value),
            ("Format",    m.format.value.upper()),
            ("Quant",     m.quantization or "none"),
            ("Memory",    f"{m.memory_requirement_mb:.0f} MB"),
            ("Devices",   ", ".join(m.supported_devices)),
            ("Runtimes",  ", ".join(m.supported_runtimes)),
            ("Local Only", "YES" if m.local_only else "NO"),
            ("License",   m.license),
        ]
        for key, val in details:
            row = QHBoxLayout()
            k = QLabel(f"{key}:")
            k.setFont(QFont("Segoe UI", 9))
            k.setStyleSheet("color: #475569; min-width: 80px;")
            v = QLabel(val)
            v.setFont(QFont("Segoe UI Mono", 9))
            v.setStyleSheet("color: #94a3b8;")
            v.setWordWrap(True)
            row.addWidget(k)
            row.addWidget(v)
            row.addStretch()
            layout.addLayout(row)

        if m.description:
            desc = QLabel(m.description)
            desc.setFont(QFont("Segoe UI", 9))
            desc.setStyleSheet("color: #334155; padding-top: 4px;")
            desc.setWordWrap(True)
            layout.addWidget(desc)


class ModelsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        QTimer.singleShot(300, self._load_models)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(16)

        title = QLabel("MODEL REGISTRY")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #818cf8; letter-spacing: 3px;")
        outer.addWidget(title)

        sub = QLabel("All registered AI models and their deployment status")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color: #475569; margin-top: -8px;")
        outer.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setSpacing(12)
        scroll.setWidget(self._container)
        outer.addWidget(scroll)

    def _load_models(self):
        try:
            from core.model_manager.manager import get_model_manager
            mm = get_model_manager()
            models = mm.list_models()

            while self._container_layout.count():
                item = self._container_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for m in models:
                present = mm.is_model_file_present(m.model_id)
                card = ModelCard(m, present)
                self._container_layout.addWidget(card)
            self._container_layout.addStretch()
        except Exception as e:
            lbl = QLabel(f"Error loading models: {e}")
            lbl.setStyleSheet("color: #ef4444;")
            self._container_layout.addWidget(lbl)
