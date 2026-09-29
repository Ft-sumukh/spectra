# SPECTRA: Snapdragon-Powered Private Multimodal AI Workspace

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20Snapdragon-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-ONNX%20%7C%20Qualcomm%20QNN-purple.svg)]()
[![Offline](https://img.shields.io/badge/Mode-100%25%20Local%20%2F%20Private-success.svg)]()

> **Built for the Snapdragon AI Lab Build & Present Challenge**  
> Tailored for Snapdragon X Elite / Plus and Snapdragon-powered HP Windows PCs.

---

## 1. What is SPECTRA?

**SPECTRA** is a local-first, privacy-preserving multimodal AI workstation built specifically for next-generation Windows AI PCs. Rather than relying on cloud APIs, SPECTRA executes AI inference directly on local silicon with transparent hardware routing:

$$\text{User Request} \longrightarrow \text{Task Detection} \longrightarrow \text{Model Selection} \longrightarrow \text{Silicon Routing (CPU / GPU / Snapdragon NPU)} \longrightarrow \text{Local Result} + \text{Telemetry}$$

The defining architectural feature of SPECTRA is its **Hardware-Aware AI Workload Router**, which dynamically matches AI tasks (Vision, Speech, Document, Text) to optimal execution providers (Qualcomm Hexagon NPU via QNN, GPU via DirectML/CUDA, CPU via ONNX Runtime).

---

## 2. Why Snapdragon?

Snapdragon X Series processors introduce dedicated Hexagon Neural Processing Units (NPUs) capable of up to 45 TOPS (Trillion Operations Per Second) at ultra-low power consumption.

Traditional AI applications either:
1. Offload private data to remote cloud servers (high latency, ongoing subscription costs, privacy exposure), or
2. Run on power-hungry discrete GPUs that exhaust laptop battery life.

SPECTRA bridges this gap by targeting the **Qualcomm QNN (Qualcomm Neural Network)** execution provider within ONNX Runtime. Lightweight edge architectures (such as MobileNetV2, MobileNet-SSD, and Quantized Transformers) execute on the Snapdragon NPU at millisecond latencies while keeping the laptop cool and battery-efficient.

---

## 3. Core Features

| Feature | Description | Status |
| :--- | :--- | :---: |
| **Hardware Detection** | Real-time discovery of CPU, GPU, NPU, RAM, OS, and PnP devices without artificial claims. | **Verified** |
| **Runtime Abstraction** | Unified manager for ONNX Runtime, Qualcomm QNN EP, PyTorch, and DirectML. | **Verified** |
| **Model Registry** | Lifecycle management for edge models with memory constraints and quantization tags. | **Verified** |
| **Workload Router** | Explainable silicon routing engine with explicit fallback logs and reasoning. | **Verified** |
| **Vision Engine** | Local image classification and object detection with MobileNetV2 (ONNX) and YOLOv8. | **Verified** |
| **Performance Lab** | Statistical latency benchmarking (Min, Max, Avg, P50, P95, P99, Throughput, Memory). | **Verified** |
| **Offline Privacy Shield**| Strict offline-first policy: zero telemetry, zero data egress, zero cloud dependencies. | **Verified** |
| **Desktop Workstation UI**| Professional PySide6 dark interface with real-time hardware telemetry and routing plans. | **Verified** |

---

## 4. Hardware Detection & Verification

SPECTRA never fabricates hardware capabilities. Below is the automated detection report generated on the current development environment:

```text
==================================================
  SPECTRA - HARDWARE DETECTION REPORT
==================================================
  OS       : Microsoft Windows 11 Home Single Language (10.0.26200)
  Hostname : SUMUKH
  CPU      : 13th Gen Intel(R) Core(TM) i5-1335U (10P / 12L cores)
  Arch     : AMD64 (x86_64)
  GPU      : Intel(R) Iris(R) Xe Graphics (2048 MB VRAM)
  NPU      : NOT DETECTED (Honest reporting: Intel host machine)
  RAM      : 15.7 GB Total / 3.5 GB Available
  Runtimes : ONNX Runtime 1.26.0 (CPUExecutionProvider, AzureExecutionProvider)
==================================================
```

*When deployed to Snapdragon HP PCs with the Qualcomm AI Stack, SPECTRA detects `QNNExecutionProvider` and routes model execution directly to the Hexagon NPU.*

---

## 5. Measured AI Performance (MobileNetV2 ONNX)

Measurements taken on local hardware via the integrated **Performance Lab**:

* **Model**: MobileNetV2 (ONNX FP32, 13.59 MB, 3.5M parameters)
* **Runtime**: ONNX Runtime 1.26.0
* **Execution Provider**: CPUExecutionProvider (Fallback on non-Snapdragon host)
* **Warmup Iterations**: 3
* **Benchmark Iterations**: 15

$$\text{Average Latency: } 6.95\text{ ms} \quad\vert\quad \text{P95 Latency: } 9.38\text{ ms} \quad\vert\quad \text{Throughput: } 143.88\text{ req/s}$$

*Memory Footprint: < 35 MB total working set during continuous inference.*

---

## 6. Architecture Overview

```text
SPECTRA/
├── app/
│   └── main.py                     # Application bootstrap & UI launcher
├── frontend/
│   ├── main_window.py              # PySide6 desktop workstation shell
│   └── pages/
│       ├── dashboard.py            # Live silicon telemetry & active models
│       ├── vision_page.py          # Multimodal image perception interface
│       ├── assistant_page.py       # Local AI assistant & execution telemetry
│       ├── performance_page.py     # Real-time latency benchmark lab
│       ├── models_page.py          # Registered model catalog & status
│       └── settings_page.py        # System configuration & flags
├── core/
│   ├── router/router.py            # Hardware-aware AI workload router
│   ├── model_manager/manager.py    # Model catalog & compatibility manager
│   └── orchestrator/               # Multimodal pipeline coordination
├── ai/
│   ├── vision/engine.py            # MobileNetV2 & YOLOv8 inference engine
│   ├── document/                   # Document intelligence pipeline
│   ├── speech/                     # Speech recognition pipeline
│   └── multimodal/                 # Cross-modal reasoning pipeline
├── runtime/
│   └── manager.py                  # ONNX, QNN, PyTorch runtime detection
├── hardware/
│   └── detector.py                 # Real CPU, GPU, NPU discovery
├── benchmarking/
│   └── benchmark.py                # Statistical latency profiling framework
├── config/
│   └── settings.py                 # Environment settings & directory paths
├── tests/
│   └── test_spectra.py             # 38 comprehensive unit & integration tests
├── scripts/
│   ├── setup.ps1                   # Automated Windows environment installer
│   ├── run.ps1                     # Desktop application launcher
│   ├── download_mobilenet.py       # MobileNetV2 ONNX acquisition utility
│   └── benchmark_cli.py            # Headless CLI benchmark runner
└── docs/                           # In-depth architectural & runtime guides
```

---

## 7. Quickstart Guide

### Prerequisites
* Windows 11 (64-bit x64 or ARM64)
* Python 3.10 to 3.14

### Automated Setup (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/your-org/spectra.git
cd spectra

# Run the automated Windows setup script
.\scripts\setup.ps1
```

### Manual Installation
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Acquire default MobileNetV2 ONNX model
python scripts/download_mobilenet.py

# 3. Run verification test suite
python tests/test_spectra.py

# 4. Launch SPECTRA UI
python app/main.py
```

### Headless Benchmarking
```powershell
python scripts/benchmark_cli.py
```

---

## 8. License

SPECTRA is released under the [Apache 2.0 License](LICENSE).
