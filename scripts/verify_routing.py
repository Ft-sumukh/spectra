"""
SPECTRA - Routing Verification Harness

Proves (or disproves) that the WorkloadRouter actually determines which
execution provider an ONNX session binds to. This is the evidence file for
the project's central claim.

Usage:
    python scripts/verify_routing.py
    python scripts/verify_routing.py --json

Exit code 0 = all routing claims hold. Non-zero = a claim failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger  # noqa: E402

log = get_logger("VERIFY")


def _matrix() -> list[dict]:
    """Build the routing evidence matrix."""
    import numpy as np
    from PIL import Image

    from ai.vision.engine import VisionEngine
    from benchmarking.benchmark import run_benchmark
    from core.execution import create_session
    from core.model_manager.manager import Modality, ModelTask
    from core.router.router import LatencyRequirement, WorkloadRequest, get_router
    from runtime.manager import ExecutionDevice

    rows: list[dict] = []
    model_path = ROOT / "models" / "mobilenetv2-7.onnx"

    # ── Silicon inventory ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SPECTRA ROUTING VERIFICATION")
    print("=" * 68)

    from hardware.detector import detect_hardware
    hw = detect_hardware()
    print(f"\n  Host CPU : {hw.cpu.name}")
    print(f"  Host GPU : {hw.gpus[0].name if hw.gpus else 'none detected'}")
    print(f"  Host NPU : {hw.npu.name if hw.npu.available else 'NOT DETECTED'}")

    import onnxruntime as ort
    available = list(ort.get_available_providers())
    print(f"\n  ONNX Runtime {ort.__version__}")
    print(f"  Providers compiled into this build: {available}")

    if not model_path.exists():
        print(f"\n  ERROR: model not found at {model_path}")
        print("  Run: python scripts/download_mobilenet.py")
        return [{"check": "model_present", "passed": False, "detail": "model missing"}]

    # ── 1. Provider resolution per requested device ───────────────────────────
    print("\n" + "-" * 68)
    print("  TEST 1: Provider resolution honours requested device")
    print("-" * 68)

    for device in ("NPU", "GPU", "CPU"):
        res = create_session(model_path, preferred_device=device)
        expect_npu = device == "NPU" and "QNNExecutionProvider" in available
        if not res.ok:
            rows.append({"check": f"session_{device}", "passed": False,
                         "detail": res.error})
            print(f"  {device}: FAILED to create session — {res.error}")
            continue

        bound = res.resolution.primary
        bound_device = res.resolution.device

        # The contract is not "the NPU always wins" — it is "if the NPU cannot
        # serve this model, we say so instead of pretending". So:
        #   device available in this ORT build -> must actually bind to it
        #   device unavailable                -> must bind CPU AND declare fallback
        ep_available = {
            "NPU": "QNNExecutionProvider" in available,
            "GPU": any(p in available for p in
                       ("CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider")),
            "CPU": True,
        }[device]

        if ep_available:
            ok = bound_device == device
            contract = "device available -> must bind to it"
        else:
            ok = (bound_device == "CPU") and res.resolution.fell_back_to_cpu
            contract = "device unavailable -> must bind CPU and declare fallback"

        print(f"\n  Requested : {device}")
        print(f"  Contract  : {contract}")
        print(f"  Requested list : {res.resolution.requested}")
        print(f"  Bound to  : {bound} (device={bound_device})")
        if res.resolution.rejected:
            print(f"  Rejected  : {res.resolution.rejected}")
        if res.resolution.fell_back_to_cpu:
            print(f"  Fallback : YES — {res.resolution.notes}")
        else:
            print(f"  Fallback : no")
        print(f"  RESULT    : {'PASS' if ok else 'FAIL'}")

        rows.append({
            "check": f"provider_resolution_{device}",
            "passed": bool(ok),
            "requested": device,
            "bound_provider": bound,
            "bound_device": bound_device,
            "fell_back": res.resolution.fell_back_to_cpu,
            "detail": "; ".join(res.resolution.notes),
        })

    # ── 2. Router plan actually influences the session ───────────────────────
    print("\n" + "-" * 68)
    print("  TEST 2: Router decision drives real execution (NPU requested)")
    print("-" * 68)

    request = WorkloadRequest(
        modality=Modality.VISION,
        task=ModelTask.IMAGE_CLASSIFICATION,
        input_type="image",
        latency_requirement=LatencyRequirement.REALTIME,
        offline_required=True,
        model_id_hint="mobilenet_v2",
        preferred_device=ExecutionDevice.NPU,
    )
    plan = get_router().route(request)
    print(f"\n  Router selected : {plan.selected_device.value}")
    print(f"  Router runtime  : {plan.selected_runtime.value}")
    print(f"  Router provider : {plan.selected_provider}")
    print(f"  Fallback used   : {plan.fallback_used}")
    if plan.fallback_reason:
        print(f"  Fallback reason : {plan.fallback_reason}")

    engine = VisionEngine()
    ok_load, msg = engine.load(model_path)
    print(f"\n  Engine load     : {ok_load} — {msg}")
    print(f"  Engine device   : {engine.device}")
    print(f"  Engine provider : {engine.provider}")
    print(f"  Requested device: {engine.requested_device}")
    print(f"  Fell back to CPU: {engine.fell_back_to_cpu}")

    consistent = (engine.requested_device == plan.selected_device.value)
    honest = True
    if engine.requested_device in ("NPU", "GPU") and engine.device == "CPU":
        honest = engine.fell_back_to_cpu  # fallback must be declared
    print(f"\n  Router/engine agree on device : {consistent}")
    print(f"  Fallback declared honestly    : {honest}")

    rows.append({
        "check": "router_drives_session",
        "passed": bool(consistent and honest and ok_load),
        "router_device": plan.selected_device.value,
        "engine_requested": engine.requested_device,
        "engine_bound": engine.device,
        "fell_back": engine.fell_back_to_cpu,
        "detail": msg,
    })

    # ── 3. Real inference + benchmark on the routed device ───────────────────
    print("\n" + "-" * 68)
    print("  TEST 3: Real inference and measured latency")
    print("-" * 68)

    sample = ROOT / "data" / "samples" / "sample_test.png"
    if not sample.exists():
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        arr = str(sample)

    result = engine.infer(arr)
    print(f"\n  Inference success : {result.success}")
    print(f"  Device            : {result.execution_device}")
    print(f"  Provider          : {result.provider}")
    print(f"  Latency           : {result.inference_time_ms:.2f} ms")
    if result.success and result.classifications:
        for c in result.classifications[:3]:
            print(f"    [{c.confidence:.2%}] {c.class_name}")
    else:
        print(f"  Error: {result.error_message}")

    rows.append({
        "check": "real_inference",
        "passed": bool(result.success and result.inference_time_ms > 0),
        "device": result.execution_device,
        "latency_ms": result.inference_time_ms,
        "detail": result.error_message or "",
    })

    bench = run_benchmark(
        inference_fn=lambda: engine.infer(arr),
        model_id="mobilenet_v2",
        model_name="MobileNetV2 (ONNX)",
        runtime="onnx",
        device=engine.device,
        provider=engine.provider,
        warmup_iterations=3,
        benchmark_iterations=15,
    )
    print(f"\n  Benchmark on bound device ({engine.device}):")
    print(f"    avg        : {bench.avg_latency_ms:.2f} ms")
    print(f"    p95        : {bench.p95_latency_ms:.2f} ms")
    print(f"    throughput : {bench.throughput_rps:.2f} req/s")

    rows.append({
        "check": "benchmark_bound_device",
        "passed": bool(bench.success and bench.avg_latency_ms > 0),
        "device": engine.device,
        "avg_ms": bench.avg_latency_ms,
        "p95_ms": bench.p95_latency_ms,
        "throughput_rps": bench.throughput_rps,
    })

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    rows = _matrix()

    passed = sum(1 for r in rows if r["passed"])
    total = len(rows)

    print("\n" + "=" * 68)
    print(f"  VERIFICATION SUMMARY: {passed}/{total} checks passed")
    print("=" * 68)
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['check']}")
        if not r["passed"] and r.get("detail"):
            print(f"         {r['detail']}")

    if args.json:
        out = ROOT / "logs" / "routing_verification.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  JSON written to {out}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
