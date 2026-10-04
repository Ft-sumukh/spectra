# SPECTRA Project Roadmap

Status reflects what is **built and verified**, not what is planned. Every
"Working" claim has a verification script that can be re-run.

## Shipped

| Phase | Milestone | Status | Evidence |
| :---: | :--- | :--- | :--- |
| 0 | Foundation - config, logging, test harness | **Working** | `tests/test_spectra.py` (38 tests) |
| 1 | Hardware & runtime detection | **Working** | `scripts/verify_routing.py` |
| 2 | Model registry + lifecycle | **Working** | 6 registered models |
| 3 | Workload router (explainable) | **Working** | `scripts/verify_routing.py` 6/6 |
| 4 | Vision - classification | **Working** | MobileNetV2, ~6-8 ms measured |
| 4b | Vision - object detection | **Working** | `scripts/verify_detection.py` 8/8 |
| 4c | Router drives real ONNX sessions | **Working** | `core/execution/session_builder.py` |
| 4d | Local LLM generation | **Working** | `scripts/verify_llm.py` 8/8 |
| 6 | Performance lab (P50/P95/P99) | **Working** | `scripts/benchmark_cli.py` |
| 11 | PySide6 workstation UI | **Working** | `python app/main.py` |
| 12a | NPU routing + QNN code path | **Written, untested on target** | needs Snapdragon hardware |
| 13 | Verification harnesses + CI | **Working** | `.github/workflows/ci.yml` |

## Not built

These are named in the original roadmap but **do not exist in the codebase**.
They are listed here so the gap is visible rather than implied.

| Phase | Milestone | Notes |
| :---: | :--- | :--- |
| 5 | Camera capture | No `ai/vision/camera.py`. Not started. |
| 7 | Document OCR / PDF | No `ai/document/`. Not started. |
| 8 | Speech-to-text | `whisper_tiny` is registered but has no loader. Not started. |
| 9 | Screen intelligence | Not started. |
| 10 | Vision-language reasoning | Not started. |
| 12b | Qualcomm AI Hub DLC export | Not started. Needs `ai-edge` tooling. |

> The original README listed `ai/document/`, `ai/speech/`, `ai/multimodal/`
> and `core/orchestrator/` in its architecture diagram. None of those
> directories exist. That diagram has been corrected.

## Next steps, in priority order

1. **Validate the NPU path on Snapdragon hardware.** This is the single
   highest-value item. `pip install onnxruntime-qnn`, run
   `scripts/verify_routing.py`, and record the real QNN numbers.
2. **Measure package power** for CPU vs NPU on identical workloads. Judges
   ask about efficiency, and self-measured watts are the credible answer.
3. **Export an INT8 DLC via Qualcomm AI Hub** so the assistant model can
   run natively on the Hexagon NPU, not just through the QNN EP.
4. **Camera pipeline** - the cheapest way to make the vision work feel live
   in a demo.
5. **Wire the LLM into vision** (ask the model about a detected scene) so
   "multimodal" becomes true rather than aspirational.

## Known limitations

- Detection is single-image, not live video.
- The LLM is text-only; no multimodal input path yet.
- `scripts/verify_llm.py` prints `is_qnn_available()` from ONNX Runtime
  GenAI, which returns `True` on non-Qualcomm hardware. It is a DLL probe,
  not NPU evidence. The script labels it as such.
- Benchmarks vary with machine load; run them on an idle machine before
  quoting numbers.
