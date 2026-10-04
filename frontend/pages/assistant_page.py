"""
SPECTRA - AI Assistant Page
Conversational interface with execution metadata display.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QFrame,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont
from utils.logger import get_logger

log = get_logger("UI")


class AssistantWorker(QObject):
    """Runs one generation on a worker thread and reports real telemetry."""

    finished = Signal(str, dict)
    error = Signal(str)

    def __init__(self, prompt: str, llm=None):
        super().__init__()
        self._prompt = prompt
        self._llm = llm

    def run(self):
        try:
            if self._llm is None:
                from ai.llm import LocalLLM
                self._llm = LocalLLM()

            if not self._llm.is_loaded:
                ok, msg = self._llm.load()
                if not ok:
                    self.error.emit(msg)
                    return

            res = self._llm.generate(self._prompt, max_tokens=512)
            if not res.success:
                self.error.emit(res.error_message or "Generation failed")
                return

            meta = {
                "model": res.model_name,
                "backend": res.backend,
                "runtime": res.backend,
                "device": res.device,
                "provider": res.provider,
                "latency_ms": res.total_ms,
                "ttft_ms": res.ttft_ms,
                "tokens": res.completion_tokens,
                "tok_per_s": res.tokens_per_second,
                "local": True,
                "requested_device": res.requested_device,
                "fell_back": res.fell_back_to_cpu,
                "notes": res.notes,
            }
            self.finished.emit(res.text, meta)
        except Exception as e:
            self.error.emit(str(e))


class AssistantPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._llm = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        title = QLabel("AI ASSISTANT")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #818cf8; letter-spacing: 3px;")
        layout.addWidget(title)

        sub = QLabel("Local AI · Hardware-aware · Private")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color: #475569; margin-top: -8px;")
        layout.addWidget(sub)

        # Chat area
        self._chat = QTextEdit()
        self._chat.setReadOnly(True)
        self._chat.setStyleSheet("""
            QTextEdit {
                background: #1a1a28; color: #e2e8f0;
                border: 1px solid #1e1e30; border-radius: 8px;
                font-family: 'Segoe UI'; font-size: 12px; padding: 16px;
            }
        """)
        layout.addWidget(self._chat)

        # Input
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask the local LLM anything...")
        self._input.setFixedHeight(42)
        self._input.setStyleSheet("""
            QLineEdit {
                background: #1a1a28; color: #e2e8f0;
                border: 1px solid #2d2d48; border-radius: 6px;
                font-size: 12px; padding: 0 14px;
            }
        """)
        self._input.returnPressed.connect(self._send)

        send_btn = QPushButton("Send")
        send_btn.setFixedSize(80, 42)
        send_btn.setStyleSheet("""
            QPushButton {
                background: #4f46e5; color: white; border: none;
                border-radius: 6px; font-weight: 600;
            }
            QPushButton:hover { background: #6366f1; }
        """)
        send_btn.clicked.connect(self._send)

        input_row.addWidget(self._input)
        input_row.addWidget(send_btn)
        layout.addLayout(input_row)

        self._chat.setText(
            "SPECTRA Local Assistant — Qwen3-0.6B (INT4)\n"
            + "-" * 46 + "\n\n"
            "Real on-device generation. Nothing leaves this machine.\n\n"
            "First run downloads the model (~500 MB):\n"
            "  python scripts/download_llm.py\n\n"
            "Then ask it anything, for example:\n"
            "  - Where is your inference running?\n"
            "  - Explain what a neural processing unit does.\n"
            "  - Write a Python function to parse an ONNX model's IO names.\n"
        )

    def _send(self):
        prompt = self._input.text().strip()
        if not prompt:
            return
        self._input.clear()

        self._chat.append(f"\nYou: {prompt}")
        self._chat.append("SPECTRA: Processing...")

        self._thread = QThread()
        self._worker = AssistantWorker(prompt, llm=self._llm)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_response)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_response(self, response: str, meta: dict):
        current = self._chat.toPlainText()
        if current.endswith("SPECTRA: Processing..."):
            current = current[:-len("SPECTRA: Processing...")].rstrip()
            self._chat.setText(current)

        self._chat.append(f"\nSPECTRA: {response}")
        self._chat.append("\n" + "-" * 46)
        self._chat.append("Execution Telemetry (measured)")
        self._chat.append(f"Model    : {meta['model']}")
        self._chat.append(f"Backend  : {meta['backend']}")
        self._chat.append(f"Device   : {meta['device']} ({meta['provider']})")
        if meta.get("requested_device") and meta["requested_device"] != meta["device"]:
            self._chat.append(f"Requested: {meta['requested_device']} — FALLBACK")
        self._chat.append(f"Tokens   : {meta.get('tokens', 0)}")
        self._chat.append(f"TTFT     : {meta.get('ttft_ms', 0):.1f} ms")
        self._chat.append(f"Throughput: {meta.get('tok_per_s', 0):.2f} tok/s")
        self._chat.append(f"Offline  : {'YES — no network calls' if meta['local'] else 'NO'}")
        if meta.get("notes"):
            self._chat.append(f"Note     : {meta['notes']}")
        self._chat.append("-" * 46 + "\n")

    def _on_error(self, msg: str):
        self._chat.append(f"\nERROR: {msg}\n")
