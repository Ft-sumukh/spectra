"""
SPECTRA - Structured Logger
Provides category-tagged logging: [SYSTEM], [HARDWARE], [RUNTIME], etc.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SpectraFormatter(logging.Formatter):
    """Custom formatter that adds category tags to log messages."""

    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        category = getattr(record, "category", "SYSTEM")
        msg = record.getMessage()
        return f"{color}[{ts}] [{category}] {record.levelname}: {msg}{self.RESET}"


class FileFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        category = getattr(record, "category", "SYSTEM")
        return f"[{ts}] [{category}] {record.levelname}: {record.getMessage()}"


class CategoryLogger:
    """Logger wrapper that injects a category tag into all records."""

    def __init__(self, base_logger: logging.Logger, category: str):
        self._logger = base_logger
        self._category = category

    def _log(self, level: int, msg: str, *args, **kwargs):
        extra = {"category": self._category}
        self._logger.log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg: str, *args, **kwargs):    self._log(logging.DEBUG, msg, *args, **kwargs)
    def info(self, msg: str, *args, **kwargs):     self._log(logging.INFO, msg, *args, **kwargs)
    def warning(self, msg: str, *args, **kwargs):  self._log(logging.WARNING, msg, *args, **kwargs)
    def error(self, msg: str, *args, **kwargs):    self._log(logging.ERROR, msg, *args, **kwargs)
    def critical(self, msg: str, *args, **kwargs): self._log(logging.CRITICAL, msg, *args, **kwargs)


_root_logger: Optional[logging.Logger] = None


def _get_log_path() -> Path:
    """Lazily import settings to avoid circular imports."""
    try:
        from config import settings
        return settings.LOG_DIRECTORY / "spectra.log"
    except Exception:
        return Path("logs") / "spectra.log"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Initialize the root logger with console + file handlers."""
    global _root_logger
    if _root_logger is not None:
        return _root_logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("spectra")
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(SpectraFormatter())
    logger.addHandler(ch)

    # File handler
    try:
        log_path = _get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(FileFormatter())
        logger.addHandler(fh)
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}", extra={"category": "SYSTEM"})

    logger.propagate = False
    _root_logger = logger
    return logger


def get_logger(category: str = "SYSTEM") -> CategoryLogger:
    """Get a category-tagged logger. Call setup_logging() first or it auto-initialises."""
    base = _root_logger or setup_logging()
    return CategoryLogger(base, category.upper())
