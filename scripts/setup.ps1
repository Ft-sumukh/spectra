# SPECTRA Windows Setup Script
# Run from PowerShell in repository root:
#   .\scripts\setup.ps1
# Add -WithLLM to also fetch the ~1.35 GB assistant model.

param(
    [switch]$WithLLM
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SPECTRA - Windows Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python 3 is not found. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Detected: $pyVersion" -ForegroundColor Green

# 2. Copy .env if not exists
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[OK] Created .env from .env.example" -ForegroundColor Green
}

# 3. Install dependencies
Write-Host "[SETUP] Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Report which execution providers are actually available.
# This matters more than anything else on a Snapdragon machine: plain
# onnxruntime has no QNN, so the NPU path stays dormant until
# onnxruntime-qnn is installed.
Write-Host "[SETUP] Execution providers in this ONNX Runtime build:" -ForegroundColor Yellow
python -c "import onnxruntime as ort; print('   ' + ', '.join(ort.get_available_providers()))"

$hasQnn = python -c "import onnxruntime as ort; print('yes' if 'QNNExecutionProvider' in ort.get_available_providers() else 'no')"
if ($hasQnn -eq "no") {
    Write-Host "[INFO] QNNExecutionProvider NOT present." -ForegroundColor Yellow
    Write-Host "[INFO] On a Snapdragon X Series PC, run: pip install onnxruntime-qnn" -ForegroundColor Yellow
    Write-Host "[INFO] Until then SPECTRA routes vision/LLM work to the CPU and" -ForegroundColor Yellow
    Write-Host "[INFO] reports that fallback honestly." -ForegroundColor Yellow
} else {
    Write-Host "[OK] QNNExecutionProvider detected - NPU routing is live." -ForegroundColor Green
}

# 5. Vision models
Write-Host "[SETUP] Downloading MobileNetV2 ONNX (classification)..." -ForegroundColor Yellow
python scripts/download_mobilenet.py

Write-Host "[SETUP] Fetching object detector + test image..." -ForegroundColor Yellow
python scripts/verify_detection.py --json

# 6. Optional local LLM
if ($WithLLM) {
    Write-Host "[SETUP] Downloading local LLM (~1.35 GB)..." -ForegroundColor Yellow
    python scripts/download_llm.py
} else {
    Write-Host "[INFO] Skipping LLM download. Re-run with -WithLLM, or:" -ForegroundColor Yellow
    Write-Host "[INFO]   python scripts/download_llm.py" -ForegroundColor Yellow
}

# 7. Prove the routing claim
Write-Host "[SETUP] Verifying that routing drives real ONNX sessions..." -ForegroundColor Yellow
python scripts/verify_routing.py --json

# 8. Run both test suites
Write-Host "[SETUP] Running test suites..." -ForegroundColor Yellow
python tests/test_spectra.py
python tests/test_routing_execution.py

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SPECTRA setup complete." -ForegroundColor Green
Write-Host "  Reports written to logs/*.json" -ForegroundColor Green
Write-Host "  Launch with: python app/main.py" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
