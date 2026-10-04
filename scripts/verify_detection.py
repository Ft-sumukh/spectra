"""
SPECTRA - Object Detection Verification

Proves the YOLO detection path returns real, correctly-labelled boxes on a
real photograph.

Usage:
    python scripts/verify_detection.py
    python scripts/verify_detection.py --json
    python scripts/verify_detection.py --conf 0.25

Exit code 0 = detection works.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger  # noqa: E402

log = get_logger("VERIFY")

MODEL_URL = "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx"
# The classic Ultralytics test image: a street scene with people, a bus and cars.
TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def ensure_model(dest: Path) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return True, f"model present ({dest.stat().st_size / 1048576:.1f} MB)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  downloading detector to {dest}...")
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "spectra/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return True, f"downloaded ({dest.stat().st_size / 1048576:.1f} MB)"
    except Exception as e:
        return False, f"download failed: {e}"


def ensure_image(dest: Path) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 1000:
        return True, "image present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  downloading test image to {dest}...")
        req = urllib.request.Request(TEST_IMAGE_URL,
                                     headers={"User-Agent": "spectra/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True, "image downloaded"
    except Exception as e:
        return False, f"image download failed: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model", default=str(ROOT / "models" / "yolov10n.onnx"))
    ap.add_argument("--image", default=str(ROOT / "data" / "samples" / "street.jpg"))
    args = ap.parse_args()

    print("\n" + "=" * 68)
    print("  SPECTRA OBJECT DETECTION VERIFICATION")
    print("=" * 68)

    rows: list[dict] = []

    model_path = Path(args.model)
    image_path = Path(args.image)

    ok, msg = ensure_model(model_path)
    print(f"\n  Detector : {msg}")
    if not ok:
        print(f"  ERROR: {msg}")
        rows.append({"check": "model_present", "passed": False, "detail": msg})
        _summarise(rows, args.json)
        return 1
    rows.append({"check": "model_present", "passed": True, "detail": msg})

    ok, msg = ensure_image(image_path)
    print(f"  Image    : {msg}")
    if not ok:
        print(f"  ERROR: {msg}")
        rows.append({"check": "image_present", "passed": False, "detail": msg})
        _summarise(rows, args.json)
        return 1
    rows.append({"check": "image_present", "passed": True, "detail": msg})

    # ── Load ────────────────────────────────────────────────────────────────
    from ai.vision.engine import VisionEngine

    engine = VisionEngine(model_id="yolov10n", conf_threshold=args.conf)
    ok, msg = engine.load(model_path)
    print(f"\n  Load     : {ok} — {msg}")
    print(f"  Task     : {engine._task}")
    print(f"  Device   : {engine.device}")
    print(f"  Provider : {engine.provider}")
    print(f"  Input    : {engine._input_size}")

    if not ok:
        rows.append({"check": "detector_loads", "passed": False, "detail": msg})
        _summarise(rows, args.json)
        return 1

    task_ok = engine._task == "object_detection"
    rows.append({"check": "detector_loads", "passed": bool(ok and task_ok),
                 "task": engine._task, "device": engine.device,
                 "detail": msg})
    print(f"  Task is object_detection : {task_ok}")

    # ── Infer ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("  DETECTION RUN")
    print("-" * 68)

    result = engine.infer(str(image_path))

    print(f"\n  Success  : {result.success}")
    print(f"  Device   : {result.execution_device} ({result.provider})")
    print(f"  Latency  : {result.inference_time_ms:.2f} ms")
    print(f"  Image    : {result.image_width}x{result.image_height}")

    if not result.success:
        print(f"  ERROR    : {result.error_message}")
        rows.append({"check": "detection_runs", "passed": False,
                     "detail": result.error_message or ""})
        _summarise(rows, args.json)
        return 1

    print(f"\n  Objects detected: {len(result.detections)}")
    for d in result.detections:
        box = [round(v) for v in d.bbox_xyxy]
        print(f"    [{d.confidence:6.2%}] {d.class_name:12s} bbox={box}")

    rows.append({
        "check": "detection_runs",
        "passed": result.success,
        "count": len(result.detections),
        "latency_ms": round(result.inference_time_ms, 2),
        "device": result.execution_device,
    })

    # ── Quality checks ──────────────────────────────────────────────────────
    n = len(result.detections)
    boxes_ok = True
    for d in result.detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            boxes_ok = False
            break
        if x1 < -1 or y1 < -1 or x2 > result.image_width + 1 or y2 > result.image_height + 1:
            boxes_ok = False
            break

    print(f"\n  Boxes well-formed  : {boxes_ok}")
    rows.append({"check": "boxes_well_formed", "passed": boxes_ok})

    named = [d for d in result.detections if not d.class_name.startswith("Class ")]
    labels_ok = len(named) == n
    print(f"  All labels named  : {labels_ok}")
    rows.append({"check": "all_labels_named", "passed": labels_ok,
                 "classes": sorted({d.class_name for d in result.detections})})

    # The street scene reliably contains people, a bus and cars.
    classes = {d.class_name for d in result.detections}
    expected = {"person", "bus", "car"} & classes
    semantic_ok = len(expected) >= 2
    print(f"  Expected classes found: {sorted(expected)}")
    print(f"  Semantically plausible : {semantic_ok}")
    rows.append({"check": "finds_expected_objects", "passed": semantic_ok,
                 "found": sorted(classes)})

    found_objects = n > 0
    print(f"\n  Detections found  : {found_objects} ({n} objects)")
    rows.append({"check": "found_objects", "passed": found_objects, "count": n})

    _summarise(rows, args.json)
    passed = sum(1 for r in rows if r["passed"])
    return 0 if passed == len(rows) else 1


def _summarise(rows: list[dict], as_json: bool) -> None:
    passed = sum(1 for r in rows if r["passed"])
    print("\n" + "=" * 68)
    print(f"  DETECTION SUMMARY: {passed}/{len(rows)} checks passed")
    print("=" * 68)
    for r in rows:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}")
        if not r["passed"] and r.get("detail"):
            print(f"         {r['detail']}")
    if as_json:
        out = ROOT / "logs" / "detection_verification.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  JSON written to {out}")


if __name__ == "__main__":
    sys.exit(main())
