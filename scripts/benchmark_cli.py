"""
SPECTRA - Command Line Benchmark Utility
Runs real inference benchmarks and outputs latency, throughput, and percentiles.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
from benchmarking.benchmark import run_benchmark, BenchmarkSession
from config import settings

def main():
    model_path = settings.MODEL_DIRECTORY / "mobilenetv2-7.onnx"
    if not model_path.exists():
        print(f"[ERROR] Model file not found at {model_path}")
        print("Run: python scripts/download_mobilenet.py")
        sys.exit(1)

    print("=" * 60)
    print("  SPECTRA CLI BENCHMARK SUITE")
    print("=" * 60)

    session = BenchmarkSession()

    # 1. CPU ExecutionProvider
    print("\n[BENCHMARK] Executing on CPU (CPUExecutionProvider)...")
    try:
        sess_cpu = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess_cpu.get_inputs()[0].name
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

        result_cpu = run_benchmark(
            inference_fn=lambda: sess_cpu.run(None, {input_name: dummy_input}),
            model_id="mobilenet_v2",
            model_name="MobileNetV2 (ONNX)",
            runtime="onnx",
            device="CPU",
            provider="CPUExecutionProvider",
            warmup_iterations=3,
            benchmark_iterations=15,
            notes="CLI benchmark run"
        )
        result_cpu.print_report()
        session.add(result_cpu)
    except Exception as e:
        print(f"[ERROR] CPU benchmark failed: {e}")

    # 2. Check for DirectML / GPU
    available_providers = ort.get_available_providers()
    if "DmlExecutionProvider" in available_providers:
        print("\n[BENCHMARK] Executing on GPU via DirectML...")
        try:
            sess_dml = ort.InferenceSession(str(model_path), providers=["DmlExecutionProvider"])
            result_dml = run_benchmark(
                inference_fn=lambda: sess_dml.run(None, {input_name: dummy_input}),
                model_id="mobilenet_v2",
                model_name="MobileNetV2 (ONNX)",
                runtime="onnx",
                device="GPU",
                provider="DmlExecutionProvider",
                warmup_iterations=3,
                benchmark_iterations=15,
                notes="DirectML GPU run"
            )
            result_dml.print_report()
            session.add(result_dml)
        except Exception as e:
            print(f"[ERROR] DirectML benchmark failed: {e}")
    else:
        print("\n[GPU] DmlExecutionProvider / CUDA not installed - skipping GPU benchmark")

    # 3. Check for Qualcomm QNN / NPU
    if "QNNExecutionProvider" in available_providers:
        print("\n[BENCHMARK] Executing on Snapdragon NPU via QNN...")
        try:
            sess_qnn = ort.InferenceSession(str(model_path), providers=["QNNExecutionProvider"])
            result_qnn = run_benchmark(
                inference_fn=lambda: sess_qnn.run(None, {input_name: dummy_input}),
                model_id="mobilenet_v2",
                model_name="MobileNetV2 (ONNX)",
                runtime="onnx",
                device="NPU",
                provider="QNNExecutionProvider",
                warmup_iterations=3,
                benchmark_iterations=15,
                notes="Snapdragon QNN NPU run"
            )
            result_qnn.print_report()
            session.add(result_qnn)
        except Exception as e:
            print(f"[ERROR] QNN benchmark failed: {e}")
    else:
        print("\n[NPU] QNNExecutionProvider NOT available on this system (requires Snapdragon hardware).")

    session.compare()

if __name__ == "__main__":
    main()
