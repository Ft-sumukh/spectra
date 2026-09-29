"""
SPECTRA - Main Application Entry Point
Launches the PySide6 desktop interface and initializes all subsystems.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Bootstrap config & logging first ─────────────────────────────────────────
from config import settings
from utils.logger import setup_logging, get_logger

setup_logging(settings.LOG_LEVEL)
log = get_logger("SYSTEM")

log.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
log.info(f"App environment: {settings.APP_ENV}")
log.info(f"Offline mode: {settings.OFFLINE_MODE}")


def check_dependencies() -> bool:
    """Check that required dependencies are available."""
    missing = []

    try:
        import PySide6
    except ImportError:
        missing.append("PySide6 (pip install PySide6)")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime (pip install onnxruntime)")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow (pip install Pillow)")

    try:
        import numpy
    except ImportError:
        missing.append("numpy (pip install numpy)")

    if missing:
        print("\n[ERROR] Missing required dependencies:")
        for dep in missing:
            print(f"  pip install {dep.split('(')[1].rstrip(')')}")
        print("\nRun: pip install -r requirements.txt\n")
        return False

    return True


def launch_ui():
    """Launch the PySide6 desktop interface."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from frontend.main_window import SpectraMainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(settings.APP_NAME)
    app.setApplicationVersion(settings.APP_VERSION)
    app.setOrganizationName("SPECTRA AI")

    # Apply dark theme
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    window = SpectraMainWindow()
    window.show()

    log.info("UI launched")
    return app.exec()


def _apply_dark_palette(app):
    """Apply a professional dark palette."""
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt

    palette = QPalette()
    # Base colors
    bg = QColor(18, 18, 24)
    surface = QColor(26, 26, 36)
    accent = QColor(99, 102, 241)   # Indigo
    text = QColor(226, 232, 240)
    text_dim = QColor(100, 116, 139)
    highlight = QColor(99, 102, 241)

    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, surface)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 42))
    palette.setColor(QPalette.ColorRole.ToolTipBase, surface)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, surface)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, text_dim)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, text_dim)

    app.setPalette(palette)


def main():
    if not check_dependencies():
        sys.exit(1)

    log.info("All dependencies verified")

    # Initialize subsystems
    log.info("Initializing hardware detection...")
    from hardware.detector import detect_hardware
    hw_profile = detect_hardware()

    log.info("Initializing runtime manager...")
    from runtime.manager import get_runtime_manager
    rt_manager = get_runtime_manager()

    log.info("Initializing model manager...")
    from core.model_manager.manager import get_model_manager
    model_manager = get_model_manager()

    log.info("Initializing workload router...")
    from core.router.router import get_router
    router = get_router()

    log.info("All subsystems initialized — launching UI")

    # Launch UI
    exit_code = launch_ui()
    log.info(f"Application exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
