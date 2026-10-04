# SPECTRA Model Catalog

Every model below is fetched by a script, never committed to the repo.

## Registered models

| Model ID | What it is | Task | Size | Works today | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mobilenet_v2` | MobileNetV2 (ONNX FP32) | Image classification | 13.6 MB | Yes | Apache-2.0 |
| `yolov8n_onnx` | YOLOv10n (ONNX FP32) | Object detection | 9.0 MB | Yes | AGPL-3.0 |
| `qwen3_1_7b_int4` | Qwen3-1.7B (INT4) | Text generation | 1.35 GB | Yes | Apache-2.0 |
| `qwen3_0_6b_int4` | Qwen3-0.6B (INT4) | Text generation | 511 MB | Yes | Apache-2.0 |
| `sentence_transformer_mini` | all-MiniLM-L6-v2 | Embedding | 90 MB | Registered only | Apache-2.0 |
| `whisper_tiny` | Whisper Tiny | Speech-to-text | 150 MB | Registered only | MIT |

"Registered only" means the model has a `ModelSpec` but no loader in the
codebase. Neither has been integrated; do not present them as working.

### Why the model_id says `yolov8n_onnx` but the weights are YOLOv10n

The ID is kept for backwards compatibility with existing references. The
weights are YOLOv10n because it ships a working end-to-end ONNX export. The
original entry pointed at `yolov8n.pt`, a PyTorch checkpoint that no ONNX
loader could ever open.

## Fetching

```powershell
python scripts/download_mobilenet.py          # 13.6 MB  - classification
python scripts/download_llm.py                # 1.35 GB  - default assistant
python scripts/download_llm.py --list         # catalog + install status
python scripts/download_llm.py --model qwen3_0_6b_int4
python scripts/verify_detection.py            # 9.0 MB  - detector + test image
```

## Choosing an LLM

`qwen3_1_7b_int4` is the default. The 0.6B model is in the catalog for
low-memory machines, but it answers questions about its own hardware
incorrectly:

```text
Qwen3-0.6B — "What is an NPU? One sentence."
  -> "NPU stands for a person, a group, or a team."      (wrong)

Qwen3-1.7B — same question
  -> "An NPU (Neural Processing Unit) is a specialized hardware component
      designed to efficiently process and accelerate neural network
      computations."                                     (correct)
```

That difference is why 1.35 GB is the default. On Snapdragon X the NPU has
the headroom for it.

## Registering a new model

Add a `ModelSpec` in `core/model_manager/manager.py`:

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

Point `model_path` at a real file relative to `models/`, or
`check_compatibility()` will report it missing — which is the intended
behaviour, and the bug the old YOLOv8 entry had.
