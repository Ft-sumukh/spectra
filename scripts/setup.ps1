# SPECTRA Windows Setup Script
# Run from PowerShell in repository root

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
python -m pip install -r requirements.txt

# 4. Download default MobileNetV2 ONNX model
Write-Host "[SETUP] Downloading default MobileNetV2 ONNX model..." -ForegroundColor Yellow
python scripts/download_mobilenet.py

# 5. Run test suite to verify
Write-Host "[SETUP] Running verification test suite..." -ForegroundColor Yellow
python tests/test_spectra.py

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SPECTRA setup complete! Run with: ./scripts/run.ps1" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
