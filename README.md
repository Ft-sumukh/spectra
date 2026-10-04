# SPECTRA

**Hardware-aware, fully offline multimodal AI workstation for Windows AI PCs.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-70%20passing-success.svg)]()
[![Runtime](https://img.shields.io/badge/runtime-ONNX%20Runtime%20%7C%20Qualcomm%20QNN-purple.svg)]()
[![Offline](https://img.shields.io/badge/network-0%20egress-success.svg)]()

> Built for the Snapdragon AI Lab Build & Present Challenge.
> Targets Snapdragon X Elite / Plus and other Windows AI PCs.

---

## What SPECTRA actually does

Everything runs on your own hardware. No API keys, no accounts, no network calls.

SPECTRA matches each AI task to the best piece of silicon in the machine,
binds the model to it, and then **reports what really happened** - including
when the hardware it wanted was not available and it had to fall back.

That last part is the design constraint. A router that silently pretends to be
using an NPU is worse than no router, because you cannot trust anything it
reports. Every routing decision in SPECTRA is checkable:

```text
Requested : NPU
Bound to  : CPUExecutionProvider (device=CPU)
Rejected  : ['QNNExecutionProvider']
FALLBACK  : Requested NPU, but ONNX Runtime bound the session to CPU.
            The model will run on CPU.
```

## The routing chain, end to end

The claim "routes workloads to NPU / GPU / CPU" only means something if the
router's decision actually reaches the inference session. In SPECTRA it does:

```text
WorkloadRequest
      |
      v
WorkloadRouter.route()           pick model + device, record every rejection
      |                          ExecutionPlan{device, provider, fallback_used, steps[]}
      v
core/execution/create_session()  build a real InferenceSession for that device
      |                          read back session.get_providers() -- the truth
      v
VisionEngine / LocalLLM          run inference, report the device it actually ran on
```

`create_session()` filters the requested provider against what ONNX Runtime
actually compiled in, always keeps `CPUExecutionProvider` last so a partially
offloadable model still runs, and reports the binding it actually received
rather than the one it asked for.

## What is genuinely implemented

| Capability | Status | Evidence |
| :--- | :--- | :--- |
| Hardware detection (CPU / GPU / NPU / RAM) | Working | `python scripts/verify_routing.py` |
| Router to ONNX session binding | Working | `scripts/verify_routing.py` (6/6) |
| Image classification (MobileNetV2 ONNX) | Working | `scripts/verify_routing.py` (6/6) |
| Object detection (YOLOv10n ONNX) | Working | `scripts/verify_detection.py` (8/8) |
| Local LLM (Qwen3 INT4 via Runtime GenAI) | Working | `scripts/verify_llm.py` (8/8) |
| Latency benchmark lab (P50/P95/P99/throughput) | Working | `scripts/benchmark_cli.py` |
| PySide6 workstation UI | Working | `python app/main.py` |
| 70 unit / integration tests | Working | both suites green |
| Snapdragon NPU (Hexagon via QNN EP) | **Needs your hardware** | see below |
| Camera / screen capture / speech pipelines | Not built | see `docs/ROADMAP.md` |

### Detection, really working

Not a placeholder - a real photo, real boxes, real confidences:

```text
$ python scripts/verify_detection.py
  Objects detected: 5
    [94.16%] bus          bbox=[9, 232, 803, 740]
    [90.90%] person       bbox=[219, 406, 345, 865]
    [90.06%] person       bbox=[49, 396, 247, 906]
    [82.59%] person       bbox=[673, 393, 810, 876]
    [52.20%] person       bbox=[0, 552, 59, 875]
```

Both YOLO export layouts are handled: raw-head `[1, 4+nc, anchors]` (YOLOv8,
with NMS applied in-process) and end-to-end `[1, N, 6]` (YOLOv10, with NMS
already baked into the graph).

### Local LLM, really working

Qwen3 INT4 running through ONNX Runtime GenAI - the same runtime that exposes
the Qualcomm QNN provider, so CPU and NPU execution share one code path.

```text
$ python scripts/verify_llm.py          # Qwen3-1.7B INT4, Intel i5-1335U CPU
  Mean throughput : 6.28 tok/s
  Mean TTFT       : 4112 ms
  Device          : CPU
```

Asked where it runs, the model answers from SPECTRA's own hardware detection
rather than guessing:

```text
Q: Where is your inference running? Name the device.
A: The inference is running on a CPU (execution provider CPUExecutionProvider).
```

Those are **CPU** numbers. The 1.7B model is roughly 2.5x slower than the 0.6B
option on this Intel CPU (17 tok/s) - which is exactly the gap an NPU is meant
to close. Measure it on Snapdragon and you should see the reverse.

> **On model choice.** Qwen3-0.6B (511 MB) stays in the catalog for
> low-memory machines, but it answers hardware questions incorrectly -
> *"What is an NPU?"* -> *"NPU stands for a person, a group, or a team."*
> Qwen3-1.7B (1.35 GB) is the default because it is the smallest size that is
> reliably competent.

## Running it

```powershell
git clone https://github.com/Ft-sumukh/spectra.git
cd spectra

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Vision models
python scripts/download_mobilenet.py     # 13.6 MB - classification
python scripts/verify_detection.py       # 9.0 MB  - detector (downloads itself)

# Local LLM (~1.35 GB) -- optional, needed for the assistant page
python scripts/download_llm.py

# Prove the claims
python scripts/verify_routing.py --json
python scripts/verify_detection.py --json
python scripts/verify_llm.py --json

# Run the app
python app/main.py
```

Tests:

```powershell
python tests/test_spectra.py            # 38 tests
python tests/test_routing_execution.py  # 32 tests
```

## Running on Snapdragon - read this

The NPU path is implemented but **untested on real Snapdragon silicon**, because
the development machine is an Intel Core i5-1335U with no NPU. Everything
below is what to do when you run it on your Snapdragon X Series machine.

Install the Qualcomm execution provider first. Plain `onnxruntime` does **not**
include QNN:

```powershell
pip install onnxruntime-qnn          # provides QNNExecutionProvider
```

Then confirm the provider is genuinely compiled in:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect 'QNNExecutionProvider' in the list
```

`python scripts/verify_routing.py` will then bind MobileNetV2 to the Hexagon
NPU and report the device it actually got. If QNN is missing you will get the
honest fallback path, not a fake NPU result.

**A trap worth knowing:** `onnxruntime_genai.is_qnn_available()` returns `True`
on non-Qualcomm hardware. It probes for a DLL, not a working NPU. Do not use
it as evidence of NPU acceleration - read `Model.device_type` or
`session.get_providers()` instead. `scripts/verify_llm.py` prints it but
explicitly labels it a DLL probe.

### Getting NPU numbers for the competition

Once QNN is live, the interesting comparison is the same model on CPU versus
NPU, with latency and package power measured for identical workloads:

```powershell
python scripts/verify_routing.py --json    # per-device binding report
python scripts/verify_llm.py --json         # tok/s on the bound device
```

For power, measure package draw with something like `powercfg /batteryreport`
or an external meter. Numbers you measured yourself are worth more than any
table you paste into a slide.

## Privacy

- No network calls in any inference path.
- No telemetry, no analytics, no crash upload.
- Model weights are fetched once from Hugging Face, then used offline.
- `OFFLINE_MODE=true` is the default in `.env.example`.

## Architecture

```text
spectra/
|-- core/
|   |-- execution/session_builder.py   router plan -> real ORT session
|   |-- router/router.py               explainable device routing
|   `-- model_manager/manager.py       model catalog + compatibility
|-- ai/
|   |-- llm/engine.py                  local LLM via ONNX Runtime GenAI
|   `-- vision/engine.py               classification + object detection
|-- runtime/manager.py                 execution provider discovery
|-- hardware/detector.py               CPU / GPU / NPU discovery
|-- benchmarking/benchmark.py          latency statistics
|-- frontend/                          PySide6 workstation UI
|-- scripts/
|   |-- verify_routing.py              proves routing drives real sessions
|   |-- verify_detection.py            proves detection works
|   |-- verify_llm.py                  proves the LLM generates
|   `-- download_llm.py                model acquisition
`-- tests/                             70 tests
```

## Documentation

| Document | Contents |
| :--- | :--- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | component design and data flow |
| [docs/PRIVACY.md](docs/PRIVACY.md) | offline guarantees |
| [docs/ROADMAP.md](docs/ROADMAP.md) | what is built and what is not |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | how latency is measured |
| [docs/MODELS.md](docs/MODELS.md) | model catalog and licenses |

## License

Apache 2.0 - see [LICENSE](LICENSE).

Model weights carry their own licenses: MobileNetV2 (Apache-2.0),
YOLOv10n (AGPL-3.0), Qwen3 (Apache-2.0).
