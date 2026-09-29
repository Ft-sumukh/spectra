# SPECTRA Installation Guide

## 1. System Requirements

* **Operating System**: Windows 11 (22H2 or newer)
* **Processor**: Snapdragon X Elite / Plus, or Intel/AMD 64-bit multi-core processor
* **Memory**: 8 GB RAM minimum (16 GB recommended)
* **Python**: 3.10 - 3.14 (64-bit)

---

## 2. Windows Quick Installation

```powershell
# 1. Clone repository
git clone https://github.com/your-org/spectra.git
cd spectra

# 2. Run PowerShell installer
.\scripts\setup.ps1
```

---

## 3. Manual Step-by-Step Setup

```powershell
# Step 1: Create and activate virtual environment (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Download MobileNetV2 ONNX model
python scripts/download_mobilenet.py

# Step 4: Verify test suite
python tests/test_spectra.py

# Step 5: Launch SPECTRA
python app/main.py
```
