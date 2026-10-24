"""Independent offline tests for the detector kit and browser-export verifier."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = [f"class-{index}" for index in range(80)]
FIXTURE_SHA256 = "525561ded68fdb70dd9f2582caa85049908a294b1bc83e11fa6a521b5c76bbf5"


def tiny_model() -> onnx.ModelProto:
    # The real ORT output depends on red pixel (x=1,y=0), whose procedural
    # fixture value is independently known to be float32(7/255).
    graph = helper.make_graph([
        helper.make_node("Flatten", ["images"], ["flattened"], axis=1),
        helper.make_node("Gather", ["flattened", "index"], ["sample"], axis=1),
        helper.make_node("Add", ["zero_logits", "sample"], ["logits"]),
        helper.make_node("Identity", ["box_values"], ["boxes"]),
    ], "detector-kit-independent-fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 300, 80]),
         helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 300, 4])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), "index"),
         numpy_helper.from_array(np.zeros((1, 300, 80), dtype=np.float32), "zero_logits"),
         numpy_helper.from_array(np.tile(np.array([0.5, 0.5, 0.2, 0.4], dtype=np.float32),
                                       (1, 300, 1)), "box_values")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def report_for(model_bytes: bytes) -> dict:
    return {
        "schemaVersion": 1,
        "artifact": {"file": "model.onnx", "sha256": hashlib.sha256(model_bytes).hexdigest(), "bytes": len(model_bytes)},
        "parityFixture": {
            "name": "dfine-rgb-pattern-v1", "phase": 0, "shape": [1, 3, 640, 640],
            "inputSha256": FIXTURE_SHA256,
            "expectedLogits": [float(np.float32(7 / 255))] * 24000,
            "expectedBoxes": np.tile(np.array([0.5, 0.5, 0.2, 0.4], dtype=np.float32), 300).tolist(),
        },
        "verification": {"tolerances": {
            "logits": {"atol": 2e-4, "rtol": 1e-4},
            "boxes": {"atol": 2e-5, "rtol": 1e-4},
        }},
    }


class Modules(unittest.TestCase):
    def module(self, relative_path: str):
        path = ROOT / relative_path
        self.assertTrue(path.is_file(), f"Missing runnable detector entry: {relative_path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def kit(self):
        return self.module("public/kit/detect.py")

    def verifier(self):
        return self.module("scripts/verify_detector.py")


class DetectorPreprocessing(Modules):
    def test_half_pixel_stretch_has_fractional_center_without_imagenet_normalization(self):
        kit = self.kit()
        pixels = np.array([[[0, 0, 0, 255], [255, 255, 255, 255]]], dtype=np.uint8)
        actual = kit.preprocess_rgba(pixels, size=3)
        self.assertEqual(actual.shape, (1, 3, 3, 3))
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(actual[0], np.tile([0.0, 0.5, 1.0], (3, 3, 1)))

    def test_partial_alpha_uses_float_white_compositing_without_byte_rounding(self):
        kit = self.kit()
        pixels = np.array([[[1, 99, 203, 128], [17, 99, 203, 0]]], dtype=np.uint8)
        actual = kit.preprocess_rgba(pixels, size=2)
        expected = np.array([(channel * 128 / 255 + 127) / 255 for channel in [1, 99, 203]], dtype=np.float32)
        np.testing.assert_array_equal(actual[0, :, 0, 0], expected)
        np.testing.assert_array_equal(actual[0, :, 0, 1], [1, 1, 1])

    def test_image_decode_applies_exif_before_recording_original_dimensions(self):
        kit = self.kit()
        with tempfile.TemporaryDirectory(prefix="ovb-detect-image-") as directory:
            path = Path(directory) / "oriented.png"
            image = Image.new("RGB", (2, 1))
            image.putdata([(255, 0, 0), (0, 0, 255)])
            exif = Image.Exif()
            exif[0x0112] = 6  # Rotate clockwise: red above blue, width=1,height=2.
            image.save(path, exif=exif)
            tensor, size = kit.preprocess_image(path, size=2)
        self.assertEqual(size, (1, 2))
        np.testing.assert_array_equal(tensor[0, 0], [[1, 1], [0, 0]])
        np.testing.assert_array_equal(tensor[0, 2], [[0, 0], [1, 1]])

    def test_invalid_pixel_shape_and_resize_size_are_rejected(self):
        kit = self.kit()
        for pixels, size in [(np.zeros((2, 3, 3), dtype=np.uint8), 640),
                             (np.zeros((0, 2, 4), dtype=np.uint8), 640),
                             (np.zeros((1, 1, 4), dtype=np.uint8), 0),
                             (np.zeros((1, 1, 4), dtype=np.uint8), True)]:
            with self.subTest(shape=pixels.shape, size=size), self.assertRaises(ValueError):
                kit.preprocess_rgba(pixels, size=size)


class DetectorPostprocessing(Modules):
    def arrays(self):
        return (np.full((1, 300, 80), -100.0, dtype=np.float32),
                np.tile(np.array([0.5, 0.25, 0.2, 0.4], dtype=np.float32), (1, 300, 1)))

    def test_sigmoid_global_ranking_keeps_two_classes_of_the_same_query(self):
        kit = self.kit()
        logits, boxes = self.arrays()
        logits[0, 2, 1] = math.log(9)
        logits[0, 0, 5] = math.log(4)
        logits[0, 2, 3] = math.log(7 / 3)
        actual = kit.postprocess_predictions(logits, boxes, LABELS, (1000, 500))
        self.assertEqual([(item["queryIndex"], item["classIndex"]) for item in actual], [(2, 1), (0, 5), (2, 3)])
        np.testing.assert_allclose([item["score"] for item in actual], [0.9, 0.8, 0.7], atol=2e-8, rtol=0)
        self.assertEqual([item["label"] for item in actual], ["class-1", "class-5", "class-3"])

    def test_boxes_scale_each_axis_and_preserve_outside_coordinates(self):
        kit = self.kit()
        logits, boxes = self.arrays()
        logits[0, 2, 1] = 2
        boxes[0, 2] = [0, 0.25, 0.4, 0.4]
        actual = kit.postprocess_predictions(logits, boxes, LABELS, (1000, 500))
        np.testing.assert_allclose(actual[0]["box"], [-200, 25, 200, 225], atol=5e-6, rtol=0)

    def test_threshold_is_inclusive_and_ties_use_flat_query_class_order(self):
        kit = self.kit()
        logits, boxes = self.arrays()
        logits[0, 0, 2] = logits[0, 1, 0] = 0
        actual = kit.postprocess_predictions(logits, boxes, LABELS, (10, 20), threshold=0.5, max_detections=1)
        self.assertEqual([(item["queryIndex"], item["classIndex"], item["score"]) for item in actual], [(0, 2, 0.5)])

    def test_sigmoid_handles_extreme_logits_without_overflow(self):
        kit = self.kit()
        logits, boxes = self.arrays()
        logits.fill(-1000)
        logits[0, 7, 4] = 1000
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            actual = kit.postprocess_predictions(logits, boxes, LABELS, (10, 20), threshold=1)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["score"], 1.0)

    def test_invalid_shapes_nonfinite_values_and_options_are_rejected(self):
        kit = self.kit()
        logits, boxes = self.arrays()
        invalid = [
            {"logits": logits[0]}, {"boxes": boxes[:, :-1]}, {"labels": LABELS[:-1]},
            {"labels": LABELS[:-1] + [""]}, {"image_size": (0, 100)},
            {"threshold": float("nan")}, {"threshold": -0.1}, {"threshold": 1.1},
            {"max_detections": 0}, {"max_detections": 1.5}, {"max_detections": True},
            {"logits": np.full_like(logits, float("nan"))},
            {"boxes": np.full_like(boxes, float("inf"))},
        ]
        for change in invalid:
            arguments = {"logits": logits, "boxes": boxes, "labels": LABELS, "image_size": (100, 200), **change}
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                kit.postprocess_predictions(**arguments)


class DetectorRuntime(Modules):
    def test_portable_kit_runs_complete_onnx_with_only_local_model_image_and_labels(self):
        kit = self.kit()
        with tempfile.TemporaryDirectory(prefix="ovb-detect-kit-") as directory:
            root = Path(directory)
            (root / "model.onnx").write_bytes(tiny_model().SerializeToString())
            (root / "labels.json").write_text(json.dumps(LABELS))
            Image.new("RGB", (10, 20), (64, 0, 0)).save(root / "image.png")
            result = kit.detect(root / "model.onnx", root / "image.png", root / "labels.json", max_detections=2)
        self.assertEqual(result["image"], {"width": 10, "height": 20})
        self.assertEqual(result["boxFormat"], "xyxy")
        self.assertEqual(len(result["detections"]), 2)
        self.assertAlmostEqual(result["detections"][0]["score"], 1 / (1 + math.exp(-64 / 255)), places=7)
        np.testing.assert_allclose(result["detections"][0]["box"], [4, 6, 6, 14], atol=2e-7, rtol=0)


class DetectorExportVerifier(Modules):
    def run_verify(self, change=None, model=None):
        verifier = self.verifier()
        model_bytes = (tiny_model() if model is None else model).SerializeToString()
        report = report_for(model_bytes)
        if change:
            change(report)
        with tempfile.TemporaryDirectory(prefix="ovb-detect-verify-") as directory:
            root = Path(directory)
            (root / "model.onnx").write_bytes(model_bytes)
            (root / "report.json").write_text(json.dumps(report))
            return verifier.verify(root / "model.onnx", root / "report.json")

    def test_reconstructed_fixture_has_fixed_bytes_and_independently_known_pixels(self):
        verifier = self.verifier()
        value = verifier.fixture_input()
        self.assertEqual(value.shape, (1, 3, 640, 640))
        self.assertEqual(value.dtype, np.dtype("float32"))
        self.assertEqual(hashlib.sha256(value.tobytes()).hexdigest(), FIXTURE_SHA256)
        np.testing.assert_array_equal(value[0, :, 0, 1], np.array([7, 2, 13], dtype=np.float32) / np.float32(255))

    def test_checker_and_real_ort_match_both_raw_outputs(self):
        result = self.run_verify()
        self.assertTrue(result["passed"])
        self.assertEqual(result["executionProvider"], "CPUExecutionProvider")
        self.assertEqual(result["inputSha256"], FIXTURE_SHA256)
        self.assertEqual(result["comparison"]["logits"]["maxAbsoluteError"], 0)
        self.assertEqual(result["comparison"]["boxes"]["maxAbsoluteError"], 0)

    def test_wrong_browser_values_fail_each_output_independently(self):
        for field, name in [("expectedLogits", "logits"), ("expectedBoxes", "boxes")]:
            with self.subTest(name=name):
                result = self.run_verify(lambda report: report["parityFixture"][field].__setitem__(0, 10))
                self.assertFalse(result["passed"])
                self.assertEqual(result["comparison"][name]["mismatchedValues"], 1)

    def test_report_integrity_shape_fixture_and_frozen_tolerances_are_enforced(self):
        changes = [
            lambda report: report.__setitem__("schemaVersion", 2),
            lambda report: report["artifact"].__setitem__("file", "fragment.onnx"),
            lambda report: report["artifact"].__setitem__("sha256", "0" * 64),
            lambda report: report["artifact"].__setitem__("bytes", 1),
            lambda report: report["parityFixture"].__setitem__("name", "unknown"),
            lambda report: report["parityFixture"].__setitem__("phase", 37),
            lambda report: report["parityFixture"].__setitem__("shape", [1, 3, 32, 32]),
            lambda report: report["parityFixture"].__setitem__("inputSha256", "0" * 64),
            lambda report: report["parityFixture"].__setitem__("expectedLogits", [0]),
            lambda report: report["parityFixture"]["expectedBoxes"].__setitem__(0, float("nan")),
            lambda report: report["verification"]["tolerances"]["logits"].__setitem__("atol", 1),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.run_verify(change)

    def test_external_data_hidden_in_a_tensor_attribute_is_rejected(self):
        model = tiny_model()
        external = TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1])
        external.data_location = TensorProto.EXTERNAL
        external.external_data.add(key="location", value="must-not-read.bin")
        model.graph.node.append(helper.make_node("Constant", [], ["unused"], value=external))
        with self.assertRaisesRegex(ValueError, "embed|external|External"):
            self.run_verify(model=model)

    def test_cli_writes_failure_evidence_and_returns_nonzero(self):
        self.verifier()  # Clear failure message before spawning a missing entry.
        with tempfile.TemporaryDirectory(prefix="ovb-detect-cli-") as directory:
            root = Path(directory)
            model_bytes = tiny_model().SerializeToString()
            (root / "model.onnx").write_bytes(model_bytes)
            report = report_for(model_bytes)
            report["parityFixture"]["expectedBoxes"][0] = 10
            (root / "report.json").write_text(json.dumps(report))
            output = root / "evidence.json"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_detector.py"),
                                     "--model", str(root / "model.onnx"), "--report", str(root / "report.json"),
                                     "--output", str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr, "")
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence, json.loads(result.stdout))
            self.assertFalse(evidence["passed"])


if __name__ == "__main__":
    unittest.main()
