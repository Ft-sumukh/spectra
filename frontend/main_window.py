"""
SPECTRA - Main Window
Professional dark desktop UI built with PySide6.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from frontend.pages.dashboard import DashboardPage
from frontend.pages.vision_page import VisionPage
from frontend.pages.performance_page import PerformancePage
from frontend.pages.models_page import ModelsPage
from frontend.pages.settings_page import SettingsPage
from frontend.pages.assistant_page import AssistantPage
from utils.logger import get_logger

log = get_logger("UI")

NAV_ITEMS = [
    ("Dashboard",   "⬡",  DashboardPage),
    ("Vision",      "👁",  VisionPage),
    ("Assistant",   "🤖", AssistantPage),
    ("Performance", "📊", PerformancePage),
    ("Models",      "🧠", ModelsPage),
    ("Settings",    "⚙",  SettingsPage),
]


class NavButton(QPushButton):
    def __init__(self, icon_char: str, label: str, parent=None):
        super().__init__(parent)
        self.setText(f"{icon_char}  {label}")
        self.setCheckable(True)
        self.setFixedHeight(46)
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(False)

    def _apply_style(self, active: bool):
        if active:
            self.setStyleSheet("""
                QPushButton {
                    background: #1e1e30;
                    color: #818cf8;
                    border: none;
                    border-left: 3px solid #6366f1;
                    border-radius: 0px;
                    padding: 0 20px;
                    text-align: left;
                    font-weight: 600;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    border-left: 3px solid transparent;
                    border-radius: 0px;
                    padding: 0 20px;
                    text-align: left;
                }
                QPushButton:hover {
                    background: #16162a;
                    color: #94a3b8;
                }
            """)

    def set_active(self, active: bool):
        self.setChecked(active)
        self._apply_style(active)


class Sidebar(QFrame):
    page_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(210)
        self.setStyleSheet("background: #12121a; border-right: 1px solid #1e1e2e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        logo_frame = QFrame()
        logo_frame.setFixedHeight(68)
        logo_frame.setStyleSheet("background: #0d0d18; border-bottom: 1px solid #1e1e2e;")
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setContentsMargins(20, 0, 0, 0)

        logo = QLabel("SPECTRA")
        logo.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        logo.setStyleSheet("color: #818cf8; letter-spacing: 4px;")
        sub = QLabel("AI Workspace")
        sub.setFont(QFont("Segoe UI", 8))
        sub.setStyleSheet("color: #475569;")
        logo_layout.addWidget(logo)
        logo_layout.addWidget(sub)

        layout.addWidget(logo_frame)

        nav_label = QLabel("  NAVIGATION")
        nav_label.setFont(QFont("Segoe UI", 7))
        nav_label.setStyleSheet("color: #334155; padding: 16px 0 8px 0; letter-spacing: 2px;")
        layout.addWidget(nav_label)

        self._buttons: list[NavButton] = []
        for i, (label, icon, _) in enumerate(NAV_ITEMS):
            btn = NavButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self.page_requested.emit(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        from config import settings
        offline_color = "#f59e0b" if settings.OFFLINE_MODE else "#6b7280"
        offline_text = "OFFLINE MODE" if settings.OFFLINE_MODE else "ONLINE MODE"
        status_bar = QLabel(f"● {offline_text}")
        status_bar.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        status_bar.setStyleSheet(f"color: {offline_color}; padding: 12px 20px; border-top: 1px solid #1e1e2e;")
        layout.addWidget(status_bar)

        self._set_active(0)

    def _set_active(self, index: int):
        for i, btn in enumerate(self._buttons):
            btn.set_active(i == index)

    def set_page(self, index: int):
        self._set_active(index)


class SpectraMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPECTRA — Snapdragon-Powered AI Workspace")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 860)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.page_requested.connect(self._navigate)
        main_layout.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background: #0f0f1a;")
        main_layout.addWidget(self._stack)

        self._pages = []
        for _, _, PageClass in NAV_ITEMS:
            try:
                page = PageClass()
                self._pages.append(page)
                self._stack.addWidget(page)
            except Exception as e:
                log.error(f"Failed to create page {PageClass.__name__}: {e}")
                placeholder = self._make_placeholder(f"Page Error: {e}")
                self._pages.append(placeholder)
                self._stack.addWidget(placeholder)

        self._stack.setCurrentIndex(0)
        log.info("Main window initialized")

    def _navigate(self, index: int):
        self._stack.setCurrentIndex(index)
        self._sidebar.set_page(index)
        log.info(f"Navigated to page {index}: {NAV_ITEMS[index][0]}")

    def _make_placeholder(self, msg: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel(msg)
        lbl.setStyleSheet("color: #ef4444; font-size: 14px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        return w
