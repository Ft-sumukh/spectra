# SPECTRA Runtime Guide: Qualcomm QNN & ONNX Execution

## 1. Overview

SPECTRA decouples models from physical hardware using an execution provider abstraction built on top of **ONNX Runtime** and native acceleration SDKs.

---

## 2. Supported Execution Providers

| Execution Provider | Target Silicon | Architecture | Notes |
| :--- | :--- | :--- | :--- |
| **`QNNExecutionProvider`** | Qualcomm Hexagon NPU | ARM64 (Snapdragon X) | Full hardware NPU acceleration via Qualcomm AI Stack. |
| **`DmlExecutionProvider`** | DirectML (GPU / NPU) | x86_64 / ARM64 | Cross-vendor acceleration on Windows DirectX 12 hardware. |
| **`CUDAExecutionProvider`**| NVIDIA Discrete GPU | x86_64 | High-throughput CUDA inference. |
| **`CPUExecutionProvider`** | Host CPU (AVX2 / NEON) | Universal | Always available fallback provider. |

---

## 3. Qualcomm Hexagon NPU Integration Path

On Snapdragon-powered HP PCs, Qualcomm's Hexagon NPU is accessed via the **Qualcomm Neural Network (QNN) SDK**:

1. **Model Format**: Standard ONNX models (e.g., `mobilenetv2-7.onnx`) or QNN pre-compiled context binaries (`.dlc`).
2. **Provider Registration**: When ONNX Runtime is compiled with QNN support (`onnxruntime-qnn`), `QNNExecutionProvider` appears in `ort.get_available_providers()`.
3. **Session Options**:
   ```python
   import onnxruntime as ort

   provider_options = [{
       "backend_path": "QnnHtp.dll",  # Hexagon Tensor Processor
       "profiling_level": "basic",
   }]
   session = ort.InferenceSession("models/mobilenetv2-7.onnx", 
                                  providers=["QNNExecutionProvider"], 
                                  provider_options=provider_options)
   ```

---

## 4. Honest Fallback Architecture

If `QNNExecutionProvider` is not present in the runtime environment:
1. `RuntimeManager` marks `ONNX Runtime (QNN/NPU)` as `[UNAVAILABLE]`.
2. `WorkloadRouter` evaluates fallback devices in priority order `[GPU -> CPU]`.
3. An explicit explanation is logged and displayed in the UI:
   ```text
   FALLBACK to CPU:
   Requested device: NPU
   Actual device: CPU
   Reason: QNN runtime unavailable on host hardware
   ```
4. No synthetic performance numbers or artificial NPU flags are ever generated.
