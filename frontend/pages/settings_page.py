"""
SPECTRA - Settings Page
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        title = QLabel("SETTINGS")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #818cf8; letter-spacing: 3px;")
        layout.addWidget(title)

        from config import settings

        def _setting_row(label: str, value: str) -> QFrame:
            frame = QFrame()
            frame.setStyleSheet(
                "QFrame { background: #1a1a28; border: 1px solid #1e1e30; border-radius: 6px; }"
            )
            row = QHBoxLayout(frame)
            row.setContentsMargins(16, 12, 16, 12)
            k = QLabel(label)
            k.setFont(QFont("Segoe UI", 10))
            k.setStyleSheet("color: #94a3b8; min-width: 200px;")
            v = QLabel(value)
            v.setFont(QFont("Segoe UI Mono", 10))
            v.setStyleSheet("color: #e2e8f0;")
            row.addWidget(k)
            row.addWidget(v)
            row.addStretch()
            return frame

        cfg_items = [
            ("App Name",            settings.APP_NAME),
            ("Version",             settings.APP_VERSION),
            ("Environment",         settings.APP_ENV),
            ("Log Level",           settings.LOG_LEVEL),
            ("Offline Mode",        str(settings.OFFLINE_MODE)),
            ("Telemetry",           str(settings.TELEMETRY_ENABLED)),
            ("Model Directory",     str(settings.MODEL_DIRECTORY)),
            ("Cache Directory",     str(settings.CACHE_DIRECTORY)),
            ("CPU Enabled",         str(settings.ENABLE_CPU)),
            ("GPU Enabled",         str(settings.ENABLE_GPU)),
            ("NPU Enabled",         str(settings.ENABLE_NPU)),
            ("Inference Timeout",   f"{settings.DEFAULT_INFERENCE_TIMEOUT_S}s"),
            ("Benchmark Warmup",    str(settings.BENCHMARK_WARMUP_ITERATIONS)),
            ("Benchmark Iters",     str(settings.BENCHMARK_ITERATIONS)),
        ]

        for k, v in cfg_items:
            layout.addWidget(_setting_row(k, v))

        note = QLabel(
            "Settings are loaded from .env file.\n"
            "Edit .env to change configuration, then restart SPECTRA."
        )
        note.setFont(QFont("Segoe UI", 9))
        note.setStyleSheet("color: #475569; padding: 12px 0;")
        layout.addWidget(note)
        layout.addStretch()
