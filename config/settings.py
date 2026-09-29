"""
SPECTRA - Application Configuration
Loads from environment variables with .env fallback.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    _env_example = Path(__file__).resolve().parent.parent / ".env.example"
    if _env_example.exists():
        load_dotenv(_env_example)

# ── Application ───────────────────────────────────────────────────────────────
APP_NAME: str = os.getenv("APP_NAME", "SPECTRA")
APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
APP_ENV: str = os.getenv("APP_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY: Path = Path(os.getenv("MODEL_DIRECTORY", str(BASE_DIR / "models")))
CACHE_DIRECTORY: Path = Path(os.getenv("CACHE_DIRECTORY", str(BASE_DIR / "data" / "cache")))
LOG_DIRECTORY: Path = BASE_DIR / "logs"

# Ensure directories exist
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

# ── Hardware Flags ────────────────────────────────────────────────────────────
ENABLE_CPU: bool = os.getenv("ENABLE_CPU", "true").lower() == "true"
ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "true").lower() == "true"
ENABLE_NPU: bool = os.getenv("ENABLE_NPU", "true").lower() == "true"

# ── Offline / Privacy ─────────────────────────────────────────────────────────
OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "true").lower() == "true"
TELEMETRY_ENABLED: bool = os.getenv("TELEMETRY_ENABLED", "false").lower() == "true"

# ── Inference Defaults ────────────────────────────────────────────────────────
DEFAULT_INFERENCE_TIMEOUT_S: float = float(os.getenv("DEFAULT_INFERENCE_TIMEOUT_S", "30.0"))
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "1"))

# ── Benchmark Defaults ────────────────────────────────────────────────────────
BENCHMARK_WARMUP_ITERATIONS: int = int(os.getenv("BENCHMARK_WARMUP_ITERATIONS", "3"))
BENCHMARK_ITERATIONS: int = int(os.getenv("BENCHMARK_ITERATIONS", "10"))

def to_dict() -> dict:
    """Return configuration as a serializable dict."""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "app_env": APP_ENV,
        "log_level": LOG_LEVEL,
        "model_directory": str(MODEL_DIRECTORY),
        "cache_directory": str(CACHE_DIRECTORY),
        "enable_cpu": ENABLE_CPU,
        "enable_gpu": ENABLE_GPU,
        "enable_npu": ENABLE_NPU,
        "offline_mode": OFFLINE_MODE,
        "telemetry_enabled": TELEMETRY_ENABLED,
        "benchmark_warmup": BENCHMARK_WARMUP_ITERATIONS,
        "benchmark_iterations": BENCHMARK_ITERATIONS,
    }
