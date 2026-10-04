"""
SPECTRA - LLM Model Acquisition

Downloads a local text-generation model in ONNX Runtime GenAI format.

Usage:
    python scripts/download_llm.py                 # default Qwen3-0.6B int4
    python scripts/download_llm.py --model qwen3_0_6b_int8
    python scripts/download_llm.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import get_logger  # noqa: E402

log = get_logger("MODEL")


def main() -> int:
    from ai.llm import CATALOG, DEFAULT_MODEL, LocalLLM

    # Default to the catalog's default so this script and the engine never
    # disagree about which model SPECTRA ships with.
    ap = argparse.ArgumentParser(description="Download a SPECTRA local LLM")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="catalog key (default: %(default)s)")
    ap.add_argument("--list", action="store_true", help="list known models")
    args = ap.parse_args()


    if args.list:
        print("\nSPECTRA Local LLM Catalog")
        print("=" * 68)
        for key, meta in CATALOG.items():
            status = "INSTALLED" if LocalLLM.is_downloaded(key) else "not installed"
            print(f"\n  {key}")
            print(f"    Name    : {meta['name']}")
            print(f"    Params  : {meta['params']} | {meta['quantization']}")
            print(f"    Size    : ~{meta['size_mb']} MB")
            print(f"    License : {meta['license']}")
            print(f"    Source  : {meta['repo']}/{meta['subfolder']}")
            print(f"    Status  : {status}")
        print()
        return 0

    if args.model not in CATALOG:
        print(f"Unknown model '{args.model}'. Known: {list(CATALOG)}")
        return 1

    meta = CATALOG[args.model]
    print("\n" + "=" * 68)
    print(f"  SPECTRA LLM ACQUISITION — {meta['name']}")
    print(f"  {meta['params']} | {meta['quantization']} | ~{meta['size_mb']} MB")
    print("=" * 68)

    ok, msg = LocalLLM.download(args.model)
    print(f"\n  {msg}\n")
    if ok:
        print("  Next: python scripts/verify_llm.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
