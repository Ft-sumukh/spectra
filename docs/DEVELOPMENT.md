# SPECTRA Development & Contribution Guidelines

## 1. Development Principles

1. **Modular Architecture**: All components must remain isolated behind interfaces (`hardware`, `runtime`, `core`, `ai`, `benchmarking`, `frontend`).
2. **Deterministic Verification**: Every new feature must be accompanied by unit and integration tests.
3. **Hardware Honesty**: Never mock or fabricate hardware availability in production code paths.

---

## 2. Running Verification

```powershell
# Run full unit and integration test suite
python tests/test_spectra.py

# Run CLI latency benchmarks
python scripts/benchmark_cli.py

# Verify hardware discovery output
python -c "from hardware.detector import detect_hardware, print_hardware_report; print_hardware_report(detect_hardware())"
```

---

## 3. Coding Standards

* PEP 8 compliant Python formatting.
* Strict type annotations (`typing`, `dataclasses`).
* Explicit structured logging using `utils/logger.py` with standard category tags (`[SYSTEM]`, `[HARDWARE]`, `[RUNTIME]`, `[MODEL]`, `[ROUTER]`, `[INFERENCE]`, `[BENCHMARK]`).
