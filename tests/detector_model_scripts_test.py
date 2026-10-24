"""Offline integrity checks for the public D-FINE preparation boundary."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ovb_detector", ROOT / "scripts/prepare_detector.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class DetectorPreparationContract(unittest.TestCase):
    def test_complete_pass_publishes_only_to_qualified_destination(self):
        def build(staging, _cache):
            (staging / "detector.onnx").write_bytes(b"qualified payload")
            (staging / "model.json").write_text(json.dumps({"source": {"modelId": PREPARE.MODEL_ID}}))
            return {"source": {"modelId": PREPARE.MODEL_ID}, "fixtures": [
                {"phase": phase, "comparisons": {name: {"passed": True} for name in PREPARE.OUTPUT_SHAPES}}
                for phase in (0, 37, 91)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(PREPARE, "_build_assets", build):
                report = PREPARE.prepare(root / "qualified", root / "cache", root / "diagnostic")
            self.assertTrue(report["passed"])
            self.assertEqual(report["status"], "transformers-to-onnx-verified")
            manifest = json.loads((root / "qualified/model.json").read_text())
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["status"], "developer-investigation")
            self.assertFalse((root / "diagnostic/model.json").exists())

    def test_roundoff_bound_uses_sum_of_absolute_products(self):
        spec = importlib.util.spec_from_file_location("ovb_diagnose", ROOT / "scripts/diagnose_detector_parity.py")
        diagnose = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnose)
        # gamma_3 * (|1| + |-2| + |3|) = 18 / (2**24 - 3).
        self.assertAlmostEqual(diagnose.float32_dot_product_bound([1, -2, 3]),
                               1.0728837978036042e-6, delta=1e-20)
        for invalid in ([], [np.inf], [np.nan]):
            with self.assertRaises(ValueError):
                diagnose.float32_dot_product_bound(invalid)

    def test_failed_numeric_gate_publishes_only_explicitly_unqualified_diagnostics(self):
        def build(staging, _cache):
            (staging / "detector.onnx").write_bytes(b"diagnostic payload")
            # The publishing boundary must derive qualification from every
            # comparison, even if a builder accidentally leaves optimistic flags.
            (staging / "model.json").write_text(json.dumps({"source": {"modelId": PREPARE.MODEL_ID}, "passed": True}))
            return {"source": {"modelId": PREPARE.MODEL_ID}, "passed": True,
                    "fixtures": [{"phase": phase, "comparisons": {
                        name: {"passed": not (phase == 0 and name == "residualCorners")}
                        for name in PREPARE.OUTPUT_SHAPES}} for phase in (0, 37, 91)]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(PREPARE, "_build_assets", build, create=True), self.assertRaisesRegex(ValueError, "qualification"):
                PREPARE.prepare(root / "qualified", root / "cache", root / "diagnostic")
            self.assertFalse((root / "qualified/model.json").exists())
            self.assertFalse((root / "qualified/detector.onnx").exists())
            result = json.loads((root / "diagnostic/model.json").read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "unqualified-intermediate-parity-failure")
            self.assertEqual((root / "diagnostic/detector.onnx").read_bytes(), b"diagnostic payload")
            failure = json.loads((root / "qualified/failure.json").read_text())
            self.assertFalse(failure["passed"])

    def test_source_failure_invalidates_old_output_and_leaves_no_usable_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("qualified", "diagnostic"):
                destination = root / name
                destination.mkdir()
                (destination / "model.json").write_text(json.dumps({"source": {"modelId": PREPARE.MODEL_ID}, "passed": True}))
                (destination / "detector.onnx").write_bytes(b"stale qualified graph")
            with patch.object(PREPARE, "_build_assets", side_effect=ValueError("SHA-256 mismatch"), create=True), self.assertRaisesRegex(ValueError, "SHA-256"):
                PREPARE.prepare(root / "qualified", root / "cache", root / "diagnostic")
            for name in ("qualified", "diagnostic"):
                self.assertFalse((root / name / "model.json").exists())
                self.assertFalse((root / name / "detector.onnx").exists())
                failure = json.loads((root / name / "failure.json").read_text())
                self.assertFalse(failure["passed"])
                self.assertEqual(failure["error"], "SHA-256 mismatch")

    def test_unowned_output_directory_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qualified"
            output.mkdir()
            (output / "user.txt").write_text("retain")
            with self.assertRaisesRegex(ValueError, "unowned"):
                PREPARE.prepare(output, root / "cache", root / "diagnostic")
            self.assertEqual((output / "user.txt").read_text(), "retain")

    def test_fp64_scalar_can_only_be_canonicalized_when_all_uses_cast_to_fp32(self):
        scalar = numpy_helper.from_array(np.array(1 / 3, np.float64))
        graph = helper.make_graph([
            helper.make_node("Constant", [], ["epsilon"], value=scalar),
            helper.make_node("Cast", ["epsilon"], ["y"], to=TensorProto.FLOAT),
        ], "scalar", [], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])])
        model = helper.make_model(graph)
        changes = PREPARE.canonicalize_fp32_constants(model)
        self.assertEqual(len(changes), 1)
        converted = numpy_helper.to_array(model.graph.node[0].attribute[0].t)
        self.assertEqual(converted.dtype, np.dtype("float32"))
        self.assertEqual(float(converted), 0.3333333432674408)
        # A FP64 computation cannot be silently rewritten as FP32.
        graph.node[1].CopyFrom(helper.make_node("Identity", ["epsilon"], ["y"]))
        unchanged = helper.make_model(graph)
        self.assertEqual(PREPARE.canonicalize_fp32_constants(unchanged), [])
        with self.assertRaises(ValueError):
            PREPARE.tensor_inventory(unchanged)

    def test_static_shape_binding_rejects_contradictions(self):
        graph = helper.make_graph([], "shape", [], [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "queries", 80])])
        model = helper.make_model(graph)
        PREPARE.bind_output_shapes(model, {"logits": [1, 300, 80]})
        self.assertEqual([d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim], [1, 300, 80])
        with self.assertRaisesRegex(ValueError, "contradicts"):
            PREPARE.bind_output_shapes(model, {"logits": [2, 300, 80]})
        with self.assertRaisesRegex(ValueError, "names"):
            PREPARE.bind_output_shapes(model, {"boxes": [1, 300, 4]})

    def test_cached_checkpoint_corruption_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            target.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                PREPARE.download_pinned("https://unused.invalid/model", target, "0" * 64, 3)
            self.assertEqual(target.read_bytes(), b"bad")

    def test_cached_checkpoint_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            target.write_bytes(b"valid bytes")
            digest = hashlib.sha256(b"valid bytes").hexdigest()
            with self.assertRaisesRegex(ValueError, "byte length"):
                PREPARE.download_pinned("https://unused.invalid/model", target, digest, 12)

    def test_parity_checks_every_element_and_near_zero_absolute_error(self):
        expected = np.array([0, 100], dtype=np.float32)
        result = PREPARE.comparison(np.array([0.00001, 100.001], dtype=np.float32), expected, "boxes")
        self.assertTrue(result["passed"])
        with self.assertRaisesRegex(ValueError, "parity"):
            PREPARE.comparison(np.array([0.00003, 100], dtype=np.float32), expected, "boxes")

    def test_parity_rejects_nonfinite_or_broadcastable_shapes(self):
        for actual in (np.array([np.nan], np.float32), np.array([np.inf], np.float32),
                       np.array([[1]], np.float32)):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                PREPARE.comparison(actual, np.array([1], np.float32), "logits")

    def test_export_rejects_external_or_non_fp32_constant_tensor(self):
        for dtype, external in ((TensorProto.FLOAT16, False), (TensorProto.FLOAT, True)):
            tensor = helper.make_tensor("hidden", dtype, [1], [1])
            if external:
                tensor.data_location = TensorProto.EXTERNAL
                tensor.external_data.add(key="location", value="hidden.bin")
            graph = helper.make_graph([helper.make_node("Constant", [], ["y"], value=tensor)],
                                      "probe", [], [helper.make_tensor_value_info("y", dtype, [1])])
            with self.subTest(dtype=dtype, external=external), self.assertRaises(ValueError):
                PREPARE.tensor_inventory(helper.make_model(graph))

    def test_export_inventory_rejects_custom_domain_and_nested_external_tensor(self):
        custom = helper.make_graph([helper.make_node("Opaque", [], ["y"], domain="unverified")],
                                  "custom", [], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        with self.assertRaisesRegex(ValueError, "domain"):
            PREPARE.tensor_inventory(helper.make_model(custom))
        hidden = numpy_helper.from_array(np.array([1], np.float32), "hidden")
        hidden.data_location = TensorProto.EXTERNAL
        hidden.external_data.add(key="location", value="nested.bin")
        branch = helper.make_graph([helper.make_node("Identity", ["hidden"], ["y"])], "branch", [],
                                   [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])], [hidden])
        outer = helper.make_graph([helper.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)],
                                  "outer", [helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        with self.assertRaisesRegex(ValueError, "External"):
            PREPARE.tensor_inventory(helper.make_model(outer))

    def test_parameter_mapping_requires_exact_head_initializer_values(self):
        source = {"model.decoder.class_embed.2.weight": np.array([[1, 2]], dtype=np.float32)}
        graph = helper.make_graph([], "head", [], [], [numpy_helper.from_array(
            np.array([[1, 2]], dtype=np.float32), "base.model.decoder.class_embed.2.weight")])
        mapping = PREPARE.map_parameters(helper.make_model(graph), source)
        self.assertEqual(mapping[0]["sourceKey"], "model.decoder.class_embed.2.weight")
        self.assertEqual(mapping[0]["transform"], "identity")
        graph.initializer[0].CopyFrom(numpy_helper.from_array(
            np.array([[1, 3]], dtype=np.float32), "base.model.decoder.class_embed.2.weight"))
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            PREPARE.map_parameters(helper.make_model(graph), source)


if __name__ == "__main__":
    unittest.main()
