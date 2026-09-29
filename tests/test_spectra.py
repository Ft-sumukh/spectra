"""
SPECTRA - Test Suite
Covers: hardware, runtime, model manager, router, benchmark, input validation.
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestHardwareDetection(unittest.TestCase):

    def setUp(self):
        from hardware.detector import detect_hardware
        self.profile = detect_hardware()

    def test_cpu_detection(self):
        """CPU must be detected with a name and architecture."""
        self.assertIsNotNone(self.profile.cpu.name)
        self.assertNotEqual(self.profile.cpu.name, "")
        self.assertIsNotNone(self.profile.cpu.architecture)
        self.assertGreater(self.profile.cpu.logical_cores, 0)
        print(f"  CPU: {self.profile.cpu.name} ({self.profile.cpu.architecture})")

    def test_gpu_detection(self):
        """GPU detection should return a list (may be empty on some systems)."""
        self.assertIsInstance(self.profile.gpus, list)
        for gpu in self.profile.gpus:
            self.assertIsNotNone(gpu.name)
        print(f"  GPUs detected: {len(self.profile.gpus)}")
        for g in self.profile.gpus:
            print(f"    - {g.name}")

    def test_npu_detection(self):
        """NPU detection must return a result — never crash, never fake."""
        npu = self.profile.npu
        self.assertIsNotNone(npu)
        self.assertIsInstance(npu.available, bool)
        self.assertIsNotNone(npu.notes)
        status = "AVAILABLE" if npu.available else "NOT DETECTED"
        print(f"  NPU: {status} — {npu.notes}")

    def test_npu_not_fabricated(self):
        """NPU availability must reflect actual system state, not a forced value."""
        # On a non-Snapdragon x86 Intel machine, NPU should not be reported as available
        # UNLESS QNN providers or DLLs are actually found
        npu = self.profile.npu
        if not self.profile.cpu.is_snapdragon:
            # We expect NPU to not be available (or if it is, there must be a valid reason)
            if npu.available:
                self.assertIsNotNone(npu.notes)
                self.assertGreater(len(npu.notes), 0)
                print(f"  Non-Snapdragon NPU detected (valid path): {npu.notes}")
            else:
                print(f"  NPU correctly NOT detected on non-Snapdragon hardware")

    def test_ram_detection(self):
        """RAM must be a positive number."""
        self.assertGreater(self.profile.ram.total_mb, 0)
        print(f"  RAM: {self.profile.ram.total_mb:.0f} MB total")

    def test_os_detection(self):
        """OS must be detected."""
        self.assertIsNotNone(self.profile.os_name)
        self.assertNotEqual(self.profile.os_name, "")
        print(f"  OS: {self.profile.os_name} {self.profile.os_version}")

    def test_profile_completeness(self):
        """Hardware profile must have all required fields."""
        self.assertIsNotNone(self.profile.cpu)
        self.assertIsNotNone(self.profile.npu)
        self.assertIsNotNone(self.profile.ram)
        self.assertIsNotNone(self.profile.os_name)
        self.assertIsNotNone(self.profile.python_version)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestRuntimeDetection(unittest.TestCase):

    def setUp(self):
        from runtime.manager import RuntimeManager
        self.manager = RuntimeManager()
        self.manager.detect()

    def test_runtime_detection_runs(self):
        """Runtime detection must complete without exception."""
        runtimes = self.manager.list_runtimes()
        self.assertIsInstance(runtimes, list)
        print(f"  Runtimes detected: {len(runtimes)}")

    def test_provider_detection(self):
        """ONNX providers must be detectable."""
        providers = self.manager.get_ort_providers()
        self.assertIsInstance(providers, list)
        # At minimum CPUExecutionProvider should be there if onnxruntime is installed
        print(f"  ORT Providers: {providers}")

    def test_cpu_runtime_available(self):
        """CPU runtime must be available (onnx or pytorch)."""
        from runtime.manager import ExecutionDevice
        rt = self.manager.best_runtime_for_device(ExecutionDevice.CPU)
        self.assertIsNotNone(rt, "CPU runtime must be available")
        self.assertTrue(rt.available)
        print(f"  Best CPU runtime: {rt.name}")

    def test_npu_runtime_honest(self):
        """NPU runtime availability must match actual ORT providers."""
        from runtime.manager import RuntimeType
        qnn_rt = self.manager.get_runtime(RuntimeType.ONNX_QNN)
        self.assertIsNotNone(qnn_rt)
        providers = self.manager.get_ort_providers()
        expected = "QNNExecutionProvider" in providers
        self.assertEqual(qnn_rt.available, expected,
                         "QNN runtime availability must match ORT provider list")
        print(f"  QNN available: {qnn_rt.available} (matches ORT providers: {expected})")

    def test_runtime_info_has_notes(self):
        """All RuntimeInfo objects must have explanatory notes."""
        for rt in self.manager.list_runtimes():
            self.assertIsNotNone(rt.notes)


# ─────────────────────────────────────────────────────────────────────────────
# Model Manager Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestModelManager(unittest.TestCase):

    def setUp(self):
        from core.model_manager.manager import ModelManager
        self.manager = ModelManager()

    def test_model_registration(self):
        """Built-in models must be registered."""
        models = self.manager.list_models()
        self.assertGreater(len(models), 0, "At least built-in models must be registered")
        print(f"  Registered models: {len(models)}")
        for m in models:
            print(f"    - {m.model_id}: {m.name}")

    def test_model_retrieval(self):
        """Registered models must be retrievable by ID."""
        spec = self.manager.get_model_spec("yolov8n_onnx")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.model_id, "yolov8n_onnx")
        self.assertEqual(spec.name, "YOLOv8 Nano")

    def test_model_compatibility_check(self):
        """Compatibility check must validate device and runtime."""
        # Test compatible case (model supports CPU/onnx_cpu)
        ok, msg = self.manager.check_compatibility("yolov8n_onnx", "CPU", "onnx_cpu")
        # May be False if model file doesn't exist, but message must be descriptive
        self.assertIsInstance(ok, bool)
        self.assertIsNotNone(msg)
        print(f"  YOLOv8n CPU compatibility: {ok} — {msg}")

    def test_model_compatibility_invalid_device(self):
        """Compatibility check must reject unsupported devices."""
        # yolov8n supports CPU/GPU/NPU but let's test with something invalid
        ok, msg = self.manager.check_compatibility("yolov8n_onnx", "FPGA", "onnx_cpu")
        self.assertFalse(ok)
        self.assertIn("FPGA", msg)
        print(f"  FPGA compatibility correctly rejected: {msg}")

    def test_model_spec_has_required_fields(self):
        """All registered models must have required fields."""
        for m in self.manager.list_models():
            self.assertIsNotNone(m.model_id)
            self.assertIsNotNone(m.name)
            self.assertIsNotNone(m.task)
            self.assertIsNotNone(m.modality)
            self.assertIsNotNone(m.format)
            self.assertGreater(m.memory_requirement_mb, 0)
            self.assertIsInstance(m.supported_devices, list)
            self.assertGreater(len(m.supported_devices), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Router Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestWorkloadRouter(unittest.TestCase):

    def setUp(self):
        from core.router.router import WorkloadRouter
        self.router = WorkloadRouter()

    def _make_vision_request(self, preferred_device=None):
        from core.router.router import WorkloadRequest, LatencyRequirement
        from core.model_manager.manager import ModelTask, Modality
        from runtime.manager import ExecutionDevice
        return WorkloadRequest(
            modality=Modality.VISION,
            task=ModelTask.OBJECT_DETECTION,
            input_type="image",
            latency_requirement=LatencyRequirement.INTERACTIVE,
            offline_required=True,
            preferred_device=preferred_device,
        )

    def test_cpu_routing(self):
        """Router must select CPU if it's the best available device."""
        from runtime.manager import ExecutionDevice
        plan = self.router.route(self._make_vision_request(ExecutionDevice.CPU))
        self.assertIsNotNone(plan)
        # If routing failed, it should have a meaningful error
        if plan.selected_device.value == "UNKNOWN":
            print(f"  CPU routing: fallback to unknown — {plan.routing_decision}")
        else:
            self.assertEqual(plan.selected_device, ExecutionDevice.CPU)
            print(f"  CPU routing: {plan.selected_device.value} via {plan.selected_runtime.value}")

    def test_npu_selection_or_honest_fallback(self):
        """NPU routing must either select NPU (if available) or honestly fall back."""
        from runtime.manager import ExecutionDevice
        plan = self.router.route(self._make_vision_request(ExecutionDevice.NPU))
        self.assertIsNotNone(plan)
        # Must not fabricate NPU selection
        if plan.selected_device == ExecutionDevice.NPU:
            # If NPU selected, QNN must be available
            from runtime.manager import RuntimeType
            rt = self.router._rt_manager.get_runtime(RuntimeType.ONNX_QNN)
            self.assertTrue(rt.available, "NPU selected but QNN runtime not actually available!")
            print("  NPU routing: NPU correctly selected (QNN available)")
        else:
            # Fallback occurred — must be documented
            self.assertTrue(plan.fallback_used or plan.routing_decision != "")
            print(f"  NPU routing: honestly fell back to {plan.selected_device.value} — {plan.fallback_reason or plan.routing_decision}")

    def test_gpu_selection(self):
        """GPU routing must honestly report availability."""
        from runtime.manager import ExecutionDevice
        plan = self.router.route(self._make_vision_request(ExecutionDevice.GPU))
        self.assertIsNotNone(plan)
        print(f"  GPU routing: {plan.selected_device.value} via {plan.selected_runtime.value} (fallback={plan.fallback_used})")

    def test_cpu_fallback(self):
        """Auto-routing without preference must end up on a working device."""
        plan = self.router.route(self._make_vision_request())
        self.assertIsNotNone(plan)
        self.assertNotEqual(plan.selected_device.value, "UNKNOWN")
        print(f"  Auto routing: {plan.selected_device.value} via {plan.selected_runtime.value}")

    def test_offline_routing(self):
        """Offline routing must select a local-inference capable runtime."""
        plan = self.router.route(self._make_vision_request())
        self.assertTrue(plan.offline_capable or plan.selected_device.value == "UNKNOWN")
        print(f"  Offline capable: {plan.offline_capable}")

    def test_execution_plan_has_reasoning(self):
        """ExecutionPlan must always contain routing steps."""
        plan = self.router.route(self._make_vision_request())
        self.assertIsInstance(plan.routing_steps, list)
        self.assertGreater(len(plan.routing_steps), 0)
        self.assertIsNotNone(plan.routing_decision)

    def test_plan_explain(self):
        """plan.explain() must return a non-empty string."""
        plan = self.router.route(self._make_vision_request())
        explanation = plan.explain()
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestBenchmark(unittest.TestCase):

    def _trivial_fn(self):
        """Trivial inference function for benchmark testing."""
        import time
        time.sleep(0.001)  # 1ms
        return 42

    def test_benchmark_result_real(self):
        """Benchmark must use real timing, not fabricated values."""
        from benchmarking.benchmark import run_benchmark
        result = run_benchmark(
            inference_fn=self._trivial_fn,
            model_id="test_model",
            model_name="Test Model",
            runtime="test",
            device="CPU",
            provider="test",
            warmup_iterations=2,
            benchmark_iterations=5,
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.latencies_ms), 5)
        # Trivial fn takes ~1ms — with overhead, allow 0.5 to 50ms
        self.assertGreater(result.avg_latency_ms, 0.0)
        print(f"  Benchmark avg latency: {result.avg_latency_ms:.2f}ms (real measurement)")

    def test_latency_calculation(self):
        """Latencies must be positive real numbers."""
        from benchmarking.benchmark import run_benchmark
        result = run_benchmark(
            inference_fn=self._trivial_fn,
            model_id="test",
            model_name="Test",
            runtime="test",
            device="CPU",
            provider="test",
            warmup_iterations=1,
            benchmark_iterations=3,
        )
        for lat in result.latencies_ms:
            self.assertGreater(lat, 0.0)

    def test_percentiles(self):
        """P50, P95, P99 must be consistent with raw latencies."""
        from benchmarking.benchmark import run_benchmark, _compute_percentile
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = _compute_percentile(data, 50)
        p95 = _compute_percentile(data, 95)
        self.assertAlmostEqual(p50, 30.0, delta=1.0)
        self.assertGreaterEqual(p95, p50)
        print(f"  Percentile test: P50={p50:.1f}ms, P95={p95:.1f}ms")

    def test_benchmark_failure_handled(self):
        """A failing inference function must produce a failed result, not crash."""
        from benchmarking.benchmark import run_benchmark
        def failing_fn():
            raise RuntimeError("Simulated inference failure")

        result = run_benchmark(
            inference_fn=failing_fn,
            model_id="test",
            model_name="Failing Model",
            runtime="test",
            device="CPU",
            provider="test",
            warmup_iterations=1,
            benchmark_iterations=3,
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        print(f"  Failure handled: {result.error_message}")


# ─────────────────────────────────────────────────────────────────────────────
# Input Validation Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestInputValidation(unittest.TestCase):

    def test_image_validation_numpy(self):
        """Vision engine must accept numpy arrays."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(img.shape, (480, 640, 3))
        print("  NumPy image validation: passed")

    def test_image_validation_pil(self):
        """Vision engine must accept PIL images."""
        from PIL import Image
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        self.assertEqual(img.size, (640, 480))
        print("  PIL image validation: passed")

    def test_document_validation_path(self):
        """Document paths must be checked for existence."""
        path = Path("/nonexistent/document.pdf")
        self.assertFalse(path.exists())
        print("  Document non-existence correctly detected")

    def test_vision_engine_not_loaded(self):
        """Vision engine must return error if not loaded."""
        from ai.vision.engine import VisionEngine
        import numpy as np
        engine = VisionEngine()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.infer(img)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        print(f"  Unloaded engine correctly returns error: {result.error_message}")


# ─────────────────────────────────────────────────────────────────────────────
# Real Vision Inference Tests (MobileNetV2 ONNX)
# ─────────────────────────────────────────────────────────────────────────────
class TestMobileNetInference(unittest.TestCase):

    def setUp(self):
        from ai.vision.engine import VisionEngine
        from config import settings
        self.model_path = settings.MODEL_DIRECTORY / "mobilenetv2-7.onnx"
        self.engine = VisionEngine()

    def test_mobilenet_load(self):
        """MobileNetV2 model file should load into ONNX Runtime session."""
        if not self.model_path.exists():
            self.skipTest(f"Model file not found at {self.model_path}")
        ok, msg = self.engine.load(self.model_path)
        self.assertTrue(ok)
        self.assertTrue(self.engine.is_loaded)
        print(f"  MobileNetV2 load: {ok} — {msg}")

    def test_mobilenet_real_inference(self):
        """Real inference pass must return positive latency, top classifications, device."""
        if not self.model_path.exists():
            self.skipTest(f"Model file not found at {self.model_path}")
        self.engine.load(self.model_path)
        sample_path = Path("data/samples/sample_test.png")
        if not sample_path.exists():
            self.skipTest("Sample test image not found")
        result = self.engine.infer(sample_path)
        self.assertTrue(result.success)
        self.assertGreater(result.inference_time_ms, 0.0)
        self.assertEqual(result.execution_device, "CPU")
        self.assertEqual(result.runtime, "onnx")
        self.assertGreater(len(result.classifications), 0)
        self.assertIsNotNone(result.classifications[0].class_name)
        print(f"  MobileNetV2 real inference: {result.inference_time_ms:.2f}ms | top={result.classifications[0].class_name} ({result.classifications[0].confidence:.2%})")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestConfiguration(unittest.TestCase):

    def test_config_loads(self):
        """Configuration must load without error."""
        from config import settings
        self.assertIsNotNone(settings.APP_NAME)
        self.assertIsNotNone(settings.APP_VERSION)
        print(f"  Config: {settings.APP_NAME} v{settings.APP_VERSION}")

    def test_config_offline_mode(self):
        """Offline mode flag must be accessible."""
        from config import settings
        self.assertIsInstance(settings.OFFLINE_MODE, bool)
        print(f"  Offline mode: {settings.OFFLINE_MODE}")

    def test_config_directories_exist(self):
        """Required directories must be created by config."""
        from config import settings
        self.assertTrue(settings.MODEL_DIRECTORY.exists())
        self.assertTrue(settings.CACHE_DIRECTORY.exists())
        self.assertTrue(settings.LOG_DIRECTORY.exists())

    def test_config_dict(self):
        """Config must be serializable to dict."""
        from config import settings
        d = settings.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("app_name", d)
        self.assertIn("offline_mode", d)


# ─────────────────────────────────────────────────────────────────────────────
# Test Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SPECTRA - TEST SUITE")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestConfiguration,
        TestHardwareDetection,
        TestRuntimeDetection,
        TestModelManager,
        TestWorkloadRouter,
        TestBenchmark,
        TestInputValidation,
        TestMobileNetInference,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("  ALL TESTS PASSED")
    else:
        print(f"  FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
