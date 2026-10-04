"""
SPECTRA - Local LLM Verification

Proves the local LLM actually generates text, reports real throughput, and
does not claim NPU execution it did not achieve.

Usage:
    python scripts/verify_llm.py
    python scripts/verify_llm.py --json
    python scripts/verify_llm.py --prompts 3

Exit code 0 = every claim holds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger  # noqa: E402

log = get_logger("VERIFY")

PROMPTS = [
    ("factual", "What is a neural processing unit? Answer in one sentence."),
    ("reasoning", "A workload needs <100ms latency and must stay private. "
                  "Should it run on a cloud GPU or a local NPU? Explain briefly."),
    ("selfaware", "What are you, and where is your inference running? "
                  "Name the hardware device explicitly."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    print("\n" + "=" * 68)
    print("  SPECTRA LOCAL LLM VERIFICATION")
    print("=" * 68)

    rows: list[dict] = []

    # ── 0. Dependency + model presence ───────────────────────────────────────
    try:
        import onnxruntime_genai as og
        print(f"\n  onnxruntime-genai : {og.__version__}")
        # These probes are DLL checks, not device checks: is_qnn_available() can
        # be True on non-Qualcomm hardware. We report them but do not treat
        # them as proof of an NPU.
        print(f"  genai DLL probes  : qnn={og.is_qnn_available()} "
              f"cuda={og.is_cuda_available()} dml={og.is_dml_available()}")
        print("  (DLL probe only — not proof of a working NPU on this host)")
        rows.append({"check": "genai_installed", "passed": True,
                     "detail": og.__version__})
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        print("  Run: pip install onnxruntime-genai")
        rows.append({"check": "genai_installed", "passed": False,
                     "detail": str(e)})
        return 1

    import onnxruntime as ort
    print(f"\n  ONNX Runtime      : {ort.__version__}")
    print(f"  ORT providers     : {list(ort.get_available_providers())}")

    from ai.llm import CATALOG, DEFAULT_MODEL, LocalLLM

    model_id = DEFAULT_MODEL
    present = LocalLLM.is_downloaded(model_id)
    print(f"\n  Model             : {CATALOG[model_id]['name']}")
    print(f"  Installed         : {present}")
    if not present:
        print("\n  ERROR: model not downloaded. Run: python scripts/download_llm.py")
        rows.append({"check": "model_present", "passed": False,
                     "detail": "not downloaded"})
        _summarise(rows, args.json)
        return 1
    rows.append({"check": "model_present", "passed": True,
                 "detail": CATALOG[model_id]["name"]})

    # ── 1. Load and report actual binding ────────────────────────────────────
    llm = LocalLLM(model_id=model_id)
    ok, msg = llm.load()
    print(f"\n  Load              : {ok} — {msg}")
    print(f"  Requested device  : {llm._state.requested_device}")
    print(f"  Bound device      : {llm.device}")
    print(f"  Bound provider    : {llm.provider}")
    print(f"  Fell back to CPU  : {llm.fell_back_to_cpu}")

    rows.append({
        "check": "llm_loads",
        "passed": bool(ok),
        "requested_device": llm._state.requested_device,
        "bound_device": llm.device,
        "bound_provider": llm.provider,
        "fell_back": llm.fell_back_to_cpu,
        "detail": msg,
    })
    if not ok:
        _summarise(rows, args.json)
        return 1

    # Honesty contract: if we asked for NPU and got CPU, that must be declared.
    honest = True
    if llm._state.requested_device == "NPU" and llm.device == "CPU":
        honest = llm.fell_back_to_cpu
    print(f"\n  Honesty contract  : {'HELD' if honest else 'VIOLATED'}")
    rows.append({"check": "fallback_declared_honestly", "passed": honest,
                 "detail": llm._state.notes})

    # ── 2. Real generation across several prompts ────────────────────────────
    print("\n" + "-" * 68)
    print("  GENERATION TESTS")
    print("-" * 68)

    n = min(args.prompts, len(PROMPTS))
    tps_list: list[float] = []
    ttft_list: list[float] = []

    for i, (kind, prompt) in enumerate(PROMPTS[:n], 1):
        print(f"\n  [{i}/{n}] ({kind}) {prompt}")
        print(f"  {'─' * 64}")
        res = llm.generate(prompt, max_tokens=args.max_tokens)

        if not res.success:
            print(f"  FAILED: {res.error_message}")
            rows.append({"check": f"generation_{kind}", "passed": False,
                         "detail": res.error_message or ""})
            continue

        print(f"\n  Response:")
        for line in res.text.splitlines() or [""]:
            print(f"    {line}")
        print(f"\n  Device      : {res.device} ({res.provider})")
        print(f"  Tokens      : {res.completion_tokens}")
        print(f"  TTFT        : {res.ttft_ms:.1f} ms")
        print(f"  Total       : {res.total_ms:.1f} ms")
        print(f"  Throughput  : {res.tokens_per_second:.2f} tok/s")

        # A real generation produces tokens and text.
        good = res.completion_tokens > 0 and len(res.text) > 0
        rows.append({
            "check": f"generation_{kind}",
            "passed": bool(good),
            "tokens": res.completion_tokens,
            "tok_per_s": round(res.tokens_per_second, 2),
            "ttft_ms": round(res.ttft_ms, 1),
            "device": res.device,
            "response_preview": res.text[:180],
        })
        tps_list.append(res.tokens_per_second)
        ttft_list.append(res.ttft_ms)

    # ── 3. Sustained throughput measurement ──────────────────────────────────
    if tps_list:
        print("\n" + "-" * 68)
        print("  SUSTAINED THROUGHPUT (measured, not estimated)")
        print("-" * 68)
        avg_tps = statistics.mean(tps_list)
        min_tps = min(tps_list)
        avg_ttft = statistics.mean(ttft_list)
        print(f"\n  Mean throughput : {avg_tps:.2f} tok/s")
        print(f"  Slowest run     : {min_tps:.2f} tok/s")
        print(f"  Mean TTFT       : {avg_ttft:.1f} ms")
        print(f"  Device          : {llm.device}")
        if llm.fell_back_to_cpu:
            print(f"  NOTE            : these numbers are CPU figures. Run on")
            print(f"                    Snapdragon with QNN for NPU numbers.")

        rows.append({
            "check": "sustained_throughput",
            "passed": avg_tps > 0,
            "mean_tok_per_s": round(avg_tps, 2),
            "min_tok_per_s": round(min_tps, 2),
            "mean_ttft_ms": round(avg_ttft, 1),
            "device": llm.device,
            "runs": len(tps_list),
        })

    _summarise(rows, args.json)
    passed = sum(1 for r in rows if r["passed"])
    return 0 if passed == len(rows) else 1


def _summarise(rows: list[dict], as_json: bool) -> None:
    passed = sum(1 for r in rows if r["passed"])
    print("\n" + "=" * 68)
    print(f"  LLM VERIFICATION SUMMARY: {passed}/{len(rows)} checks passed")
    print("=" * 68)
    for r in rows:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}")
        if not r["passed"] and r.get("detail"):
            print(f"         {r['detail']}")

    if as_json:
        out = ROOT / "logs" / "llm_verification.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  JSON written to {out}")


if __name__ == "__main__":
    sys.exit(main())
