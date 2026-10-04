"""
SPECTRA — Tests for the routing, execution, detection and LLM layers.

These cover code added after the original suite: the session builder that
turns router decisions into real ONNX sessions, the YOLO decoding paths, the
COCO label table, and the local LLM's reasoning/fallback handling.

Runs standalone:  python tests/test_routing_execution.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Session builder — the layer that makes routing real
# ─────────────────────────────────────────────────────────────────────────────
class TestSessionBuilder(unittest.TestCase):

    def test_cpu_request_yields_cpu(self):
        from core.execution import build_providers
        res = build_providers("CPU", available_providers=["CPUExecutionProvider"])
        self.assertIn("CPUExecutionProvider", res.requested)
        self.assertEqual(res.device, "CPU") if hasattr(res, "device") else None

    def test_npu_request_without_qnn_declares_fallback(self):
        """The honesty contract: ask NPU, get CPU, say so."""
        from core.execution import build_providers
        res = build_providers("NPU", available_providers=["CPUExecutionProvider"])
        self.assertEqual(res.requested, ["CPUExecutionProvider"])
        self.assertIn("QNNExecutionProvider", res.rejected)
        self.assertTrue(res.fell_back_to_cpu)
        self.assertTrue(res.notes, "a fallback must carry an explanation")

    def test_npu_request_with_qnn_keeps_qnn_first(self):
        from core.execution import build_providers
        res = build_providers("NPU",
                              available_providers=["QNNExecutionProvider",
                                                   "CPUExecutionProvider"])
        self.assertEqual(res.requested[0], "QNNExecutionProvider")
        # CPU must still be last so partial offload can run.
        self.assertEqual(res.requested[-1], "CPUExecutionProvider")
        self.assertFalse(res.fell_back_to_cpu)

    def test_cpu_always_present_as_safety_net(self):
        from core.execution import build_providers
        for dev in ("NPU", "GPU", "CPU"):
            res = build_providers(dev,
                                  available_providers=["QNNExecutionProvider",
                                                       "DmlExecutionProvider"])
            self.assertIn("CPUExecutionProvider", res.requested,
                          f"CPU must never be dropped from {dev}")

    def test_unknown_provider_maps_to_unknown(self):
        from core.execution import provider_device_map
        self.assertEqual(provider_device_map("QNNExecutionProvider"), "NPU")
        self.assertEqual(provider_device_map("DmlExecutionProvider"), "GPU")
        self.assertEqual(provider_device_map("SomethingElse"), "UNKNOWN")

    def test_missing_model_returns_error_not_exception(self):
        from core.execution import create_session
        out = create_session(Path("does/not/exist.onnx"), preferred_device="CPU")
        self.assertFalse(out.ok)
        self.assertIn("not found", out.error)

    def test_real_session_reports_actual_binding(self):
        """Bind a real model and confirm we report what ORT granted."""
        from core.execution import create_session
        model = Path("models/mobilenetv2-7.onnx")
        if not model.exists():
            self.skipTest("mobilenetv2-7.onnx not downloaded")
        out = create_session(model, preferred_device="NPU")
        self.assertTrue(out.ok, out.error)
        self.assertIn(out.resolution.primary, out.resolution.granted)
        # NPU was requested; on this host it must be a declared fallback.
        self.assertTrue(out.resolution.fell_back_to_cpu)
        self.assertEqual(out.device, "CPU")


# ─────────────────────────────────────────────────────────────────────────────
# COCO labels
# ─────────────────────────────────────────────────────────────────────────────
class TestCocoLabels(unittest.TestCase):

    def test_exactly_80_classes(self):
        from utils.coco_labels import COCO_CLASSES, NUM_COCO_CLASSES
        self.assertEqual(NUM_COCO_CLASSES, 80)
        self.assertEqual(len(COCO_CLASSES), 80)

    def test_canonical_indices(self):
        """A wrong index silently mislabels every detection."""
        from utils.coco_labels import COCO_CLASSES
        self.assertEqual(COCO_CLASSES[0], "person")
        self.assertEqual(COCO_CLASSES[2], "car")
        self.assertEqual(COCO_CLASSES[5], "bus")
        self.assertEqual(COCO_CLASSES[79], "toothbrush")

    def test_no_duplicates(self):
        from utils.coco_labels import COCO_CLASSES
        self.assertEqual(len(set(COCO_CLASSES)), len(COCO_CLASSES))


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────
class TestNMS(unittest.TestCase):

    def test_suppresses_overlapping_boxes(self):
        from ai.vision.engine import _nms
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]],
                         dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = _nms(boxes, scores, iou_threshold=0.45)
        self.assertEqual(len(keep), 2, "the overlapping pair must collapse to one")
        self.assertIn(2, keep, "the distant box must survive")

    def test_empty_input(self):
        from ai.vision.engine import _nms
        self.assertEqual(_nms(np.zeros((0, 4), dtype=np.float32),
                              np.zeros((0,), dtype=np.float32), 0.45), [])

    def test_sorted_by_score(self):
        from ai.vision.engine import _nms
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
        scores = np.array([0.2, 0.95], dtype=np.float32)
        keep = _nms(boxes, scores, 0.45)
        self.assertEqual(keep[0], 1, "highest score must be kept first")


# ─────────────────────────────────────────────────────────────────────────────
# Vision engine — routing wiring and detection decoding
# ─────────────────────────────────────────────────────────────────────────────
class TestVisionEngineRouting(unittest.TestCase):

    def test_unloaded_engine_reports_error(self):
        from ai.vision.engine import VisionEngine
        res = VisionEngine().infer(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertFalse(res.success)
        self.assertIsNotNone(res.error_message)

    def test_engine_exposes_routing_state(self):
        from ai.vision.engine import VisionEngine
        e = VisionEngine()
        for attr in ("fell_back_to_cpu", "fallback_reason", "requested_device",
                     "route_plan", "provider", "device", "routing_notes"):
            self.assertTrue(hasattr(e, attr), f"missing {attr}")

    def test_load_records_requested_and_bound_device(self):
        from ai.vision.engine import VisionEngine
        model = Path("models/mobilenetv2-7.onnx")
        if not model.exists():
            self.skipTest("mobilenetv2-7.onnx not downloaded")
        e = VisionEngine()
        ok, _ = e.load(model)
        self.assertTrue(ok)
        self.assertIn(e.requested_device, ("NPU", "GPU", "CPU"))
        self.assertTrue(e.device in ("CPU", "GPU", "NPU"))
        # If they differ, the difference must be declared.
        if e.requested_device != e.device:
            self.assertTrue(e.fell_back_to_cpu)

    def test_classification_result_reports_device(self):
        from ai.vision.engine import VisionEngine
        model = Path("models/mobilenetv2-7.onnx")
        if not model.exists():
            self.skipTest("mobilenetv2-7.onnx not downloaded")
        e = VisionEngine()
        e.load(model)
        sample = Path("data/samples/sample_test.png")
        res = e.infer(str(sample) if sample.exists() else np.zeros((224, 224, 3), np.uint8))
        self.assertTrue(res.success, res.error_message)
        self.assertEqual(res.task, "classification")
        self.assertGreater(res.inference_time_ms, 0.0)
        self.assertGreater(len(res.classifications), 0)


class TestDetectionDecoding(unittest.TestCase):

    def _engine(self):
        from ai.vision.engine import VisionEngine
        return VisionEngine(conf_threshold=0.25)

    def test_end_to_end_format_decodes(self):
        """[1, N, 6] export: x1,y1,x2,y2,conf,class."""
        e = self._engine()
        out = np.array([[[10.0, 20.0, 110.0, 220.0, 0.9, 2.0],
                        [0.0, 0.0, 50.0, 50.0, 0.4, 0.0]]], dtype=np.float32)
        dets = e._postprocess_detection(out, 1.0, (0.0, 0.0), 640, 480)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0].class_name, "car")
        self.assertAlmostEqual(dets[0].confidence, 0.9, places=3)

    def test_end_to_end_respects_threshold(self):
        e = self._engine()
        out = np.array([[[0.0, 0.0, 10.0, 10.0, 0.10, 0.0]]], dtype=np.float32)
        dets = e._postprocess_detection(out, 1.0, (0.0, 0.0), 640, 480,
                                        conf_threshold=0.35)
        self.assertEqual(dets, [])

    def test_raw_head_format_decodes_with_nms(self):
        """[1, 4+80, anchors] export — YOLOv8 style."""
        e = self._engine()
        nc, anchors = 80, 4
        raw = np.zeros((1, 4 + nc, anchors), dtype=np.float32)
        # Two anchors for the same class, heavily overlapping -> NMS to one.
        raw[0, :4, 0] = [320, 240, 200, 200]   # cx, cy, w, h
        raw[0, 4 + 5, 0] = 0.9                  # class 5 = bus
        raw[0, :4, 1] = [325, 245, 200, 200]
        raw[0, 4 + 5, 1] = 0.6
        dets = e._postprocess_detection(raw, 1.0, (0.0, 0.0), 640, 480)
        self.assertEqual(len(dets), 1, "overlapping same-class boxes must merge")
        self.assertEqual(dets[0].class_name, "bus")

    def test_raw_head_thresholds_out_low_scores(self):
        e = self._engine()
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, :4, 0] = [100, 100, 50, 50]
        raw[0, 4 + 0, 0] = 0.01
        dets = e._postprocess_detection(raw, 1.0, (0.0, 0.0), 640, 480,
                                        conf_threshold=0.35)
        self.assertEqual(dets, [])

    def test_letterbox_padding_is_undone(self):
        """Boxes must map back to original pixels, not letterboxed ones."""
        e = self._engine()
        out = np.array([[[100.0, 100.0, 200.0, 200.0, 0.9, 0.0]]], dtype=np.float32)
        # scale=0.5, pad=(80, 80): letterboxed 100 -> (100-80)/0.5 = 40
        dets = e._postprocess_detection(out, 0.5, (80.0, 80.0), 400, 400)
        self.assertAlmostEqual(dets[0].bbox_xyxy[0], 40.0, places=1)
        self.assertAlmostEqual(dets[0].bbox_xyxy[2], 240.0, places=1)

    def test_malformed_output_raises_honestly(self):
        e = self._engine()
        with self.assertRaises(ValueError):
            e._postprocess_detection(np.zeros((1, 2, 2), dtype=np.float32),
                                     1.0, (0.0, 0.0), 640, 480)


# ─────────────────────────────────────────────────────────────────────────────
# Local LLM — reasoning stripping and fallback reporting
# ─────────────────────────────────────────────────────────────────────────────
class TestLocalLLMHelpers(unittest.TestCase):

    def test_reasoning_is_split_out(self):
        from ai.llm.engine import _split_reasoning
        raw = "<think>weighing options</think>The answer is CPU."
        reasoning, answer = _split_reasoning(raw)
        self.assertIn("weighing options", reasoning)
        self.assertEqual(answer, "The answer is CPU.")

    def test_plain_answer_has_no_reasoning(self):
        from ai.llm.engine import _split_reasoning
        reasoning, answer = _split_reasoning("Just an answer.")
        self.assertEqual(reasoning, "")
        self.assertEqual(answer, "Just an answer.")

    def test_truncated_reasoning_is_recovered(self):
        from ai.llm.engine import _split_reasoning
        raw = "<think>started reasoning but never finished"
        reasoning, answer = _split_reasoning(raw)
        self.assertTrue(reasoning)
        self.assertEqual(answer, "")

    def test_catalog_entries_are_wellformed(self):
        from ai.llm import CATALOG
        self.assertTrue(CATALOG)
        for key, meta in CATALOG.items():
            for field in ("name", "repo", "subfolder", "size_mb", "params",
                          "quantization", "license"):
                self.assertIn(field, meta, f"{key} missing {field}")
            self.assertTrue(meta["repo"].count("/") >= 1,
                            f"{key} repo must be org/name")

    def test_generation_without_model_reports_error(self):
        from ai.llm import LocalLLM
        llm = LocalLLM()
        res = llm.generate("hello", max_tokens=8)
        if not res.success:
            self.assertIsNotNone(res.error_message)
            self.assertEqual(res.prompt, "hello")

    def test_llm_result_serialises(self):
        from ai.llm.engine import GenerationResult
        r = GenerationResult(success=True, text="hi", completion_tokens=2)
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertIn("tokens_per_second", d)
        self.assertIn("fell_back_to_cpu", d)

    def test_system_prompt_states_real_device(self):
        """The model must be told where it actually runs, not left to guess."""
        from ai.llm import LocalLLM
        sp = LocalLLM().system_prompt()
        self.assertIn("CPU", sp)  # default unbound state
        self.assertIn("SPECTRA", sp)


# ─────────────────────────────────────────────────────────────────────────────
# Router fixes
# ─────────────────────────────────────────────────────────────────────────────
class TestRouterFallbackHonesty(unittest.TestCase):

    def test_auto_route_fallback_has_a_reason(self):
        """Regression: the router used to log 'Fallback used: None'."""
        from core.model_manager.manager import Modality, ModelTask
        from core.router.router import (
            LatencyRequirement, WorkloadRequest, get_router,
        )
        req = WorkloadRequest(
            modality=Modality.VISION,
            task=ModelTask.IMAGE_CLASSIFICATION,
            input_type="image",
            latency_requirement=LatencyRequirement.INTERACTIVE,
            offline_required=True,
            model_id_hint="mobilenet_v2",
        )
        plan = get_router().route(req)
        self.assertNotEqual(plan.selected_device.value, "UNKNOWN")
        if plan.fallback_used:
            self.assertIsNotNone(plan.fallback_reason)
            self.assertNotEqual(plan.fallback_reason.strip(), "",
                                "a fallback must explain itself")
        self.assertTrue(plan.routing_steps)

    def test_generation_task_routes(self):
        """A GENERATION task must now be routable (needs a registered LLM)."""
        from core.model_manager.manager import Modality, ModelTask
        from core.router.router import (
            LatencyRequirement, WorkloadRequest, get_router,
        )
        req = WorkloadRequest(
            modality=Modality.TEXT,
            task=ModelTask.GENERATION,
            input_type="text",
            latency_requirement=LatencyRequirement.INTERACTIVE,
            offline_required=True,
        )
        plan = get_router().route(req)
        self.assertNotEqual(plan.selected_model_id, "NONE")


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  SPECTRA — ROUTING / EXECUTION / DETECTION / LLM TESTS")
    print("=" * 62 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in (TestSessionBuilder, TestCocoLabels, TestNMS,
               TestVisionEngineRouting, TestDetectionDecoding,
               TestLocalLLMHelpers, TestRouterFallbackHonesty):
        suite.addTests(loader.loadTestsFromTestCase(tc))

    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)

    print("\n" + "=" * 62)
    if result.wasSuccessful():
        print("  ALL TESTS PASSED")
    else:
        print(f"  FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 62 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
