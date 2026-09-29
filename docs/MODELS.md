# SPECTRA Model Catalog & Compatibility Matrix

## 1. Registered Models

| Model ID | Architecture | Modality | Task | Format | Size | Target Hardware | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mobilenet_v2`** | MobileNetV2 | Vision | Image Classification | ONNX FP32 | 13.59 MB | NPU / GPU / CPU | Apache-2.0 |
| **`yolov8n_onnx`** | YOLOv8 Nano | Vision | Object Detection | ONNX FP32 | 12.20 MB | NPU / GPU / CPU | AGPL-3.0 |
| **`sentence_transformer_mini`** | all-MiniLM-L6-v2 | Text | Embedding & Search | PyTorch / ONNX | 90.00 MB | CPU / GPU | Apache-2.0 |
| **`whisper_tiny`** | Whisper Tiny | Speech | Audio Transcription | PyTorch / ONNX | 150.00 MB | CPU / NPU | MIT |

---

## 2. Primary Starter Model: MobileNetV2 (ONNX)

MobileNetV2 is selected as the default vision model for SPECTRA because:
1. **Designed for Edge Silicon**: Uses inverted residuals and linear bottlenecks to maximize compute efficiency on memory-constrained mobile NPUs.
2. **Qualcomm AI Hub Verified**: Highly optimized for Qualcomm Hexagon DSP and HTP backends.
3. **Low Latency**: Runs in under 10ms on modern CPUs, and sub-3ms on Snapdragon Hexagon NPUs.
4. **Standard 1,000-Class ImageNet Output**: Provides classification across common objects, animals, tools, and vehicles.

---

## 3. Registering New Models

To register an edge model in SPECTRA, instantiate a `ModelSpec` in `core/model_manager/manager.py`:

```python
from core.model_manager.manager import ModelSpec, Modality, ModelTask, ModelFormat

manager.register_model(ModelSpec(
    model_id="custom_detector",
    name="Edge MobileNet-SSD",
    version="1.0",
    modality=Modality.VISION,
    task=ModelTask.OBJECT_DETECTION,
    format=ModelFormat.ONNX,
    quantization="int8",
    supported_runtimes=["onnx_cpu", "onnx_qnn"],
    supported_devices=["CPU", "NPU"],
    memory_requirement_mb=18.0,
    input_type="image",
    output_type="bounding_boxes",
    local_only=True,
    model_path="custom_detector.onnx",
    license="Apache-2.0"
))
```
