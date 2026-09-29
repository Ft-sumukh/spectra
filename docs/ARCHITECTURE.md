# SPECTRA Architecture Specification

## 1. Architectural Philosophy

SPECTRA is built on four core tenets:
1. **Local-First Processing**: Sensitive multimodal data (vision, documents, audio, screen) never leaves the host computer.
2. **Hardware-Aware AI Routing**: Workloads are dynamically matched to silicon (CPU, GPU, NPU) based on latency constraints, memory budgets, and available execution providers.
3. **Transparent Fallbacks**: The system never fakes AI acceleration. If an NPU runtime is unavailable, SPECTRA logs an explicit fallback notice and records the diagnostic cause.
4. **Model & Runtime Independence**: AI pipelines are isolated behind standardized interfaces (`VisionEngine`, `RuntimeManager`, `WorkloadRouter`).

---

## 2. Core Routing Pipeline

The central innovation in SPECTRA is the **WorkloadRouter** (`core/router/router.py`). When a task request arrives:

```text
+-------------------------------------------------------+
|                   WorkloadRequest                     |
|  - Task: Image Classification / Object Detection      |
|  - Modality: Vision                                   |
|  - Latency Budget: Real-time (<50ms) / Interactive    |
|  - Preferred Silicon: NPU (Qualcomm Hexagon)          |
|  - Offline Requirement: Strict Local                  |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                 Model Compatibility                   |
|  - Matches task to candidate models in registry       |
|  - Verifies format (ONNX, QNN DLC, GGUF)              |
|  - Checks memory requirement vs available RAM         |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                Hardware & Runtime Match               |
|  - Evaluates device priority: [NPU -> GPU -> CPU]     |
|  - Queries RuntimeManager for available providers     |
|  - Inspects QNNExecutionProvider / DirectML / CPU EP  |
+-------------------------------------------------------+
                           |
            +--------------+--------------+
            |                             |
      (NPU Available)              (NPU Not Found)
            |                             |
            v                             v
+-----------------------+   +---------------------------+
|    Execute on NPU     |   |    Transparent Fallback   |
| Model: MobileNetV2    |   | Target: CPUExecutionProv. |
| Provider: QNN-EP      |   | Reason: QNN EP missing    |
| Status: Hardware Accel|   | Status: Fallback Active   |
+-----------------------+   +---------------------------+
```

---

## 3. Execution Plan Schema

Every routing decision produces an `ExecutionPlan` object containing:
* `selected_model_id`: Identifier of registered model (e.g. `mobilenet_v2`).
* `selected_runtime`: Inference runtime type (`onnx_cpu`, `onnx_qnn`, etc.).
* `selected_device`: Target silicon (`CPU`, `GPU`, `NPU`).
* `selected_provider`: Specific provider string passed to ONNX Runtime.
* `fallback_used`: Boolean indicating whether a fallback occurred.
* `fallback_reason`: Diagnostic description of why the primary target was bypassed.
* `routing_steps`: Sequential audit log of the router's decision logic.

---

## 4. Subsystem Isolation

* **`hardware/`**: Low-level device discovery querying WMI, OS system libraries, and Windows PnP. Completely independent of AI libraries.
* **`runtime/`**: Runtime discovery testing actual driver and library availability (`onnxruntime.get_available_providers()`, PyTorch CUDA, QNN DLLs).
* **`core/model_manager/`**: Model metadata catalog, download helpers, and integrity validation.
* **`ai/`**: High-level perception and reasoning engines executing preprocessing, inference, and postprocessing.
* **`benchmarking/`**: Statistically sound profiling framework computing mean, min, max, P50, P95, and P99 latencies without synthetic fabrication.
* **`frontend/`**: PySide6 workstation interface receiving asynchronous updates via worker threads.
