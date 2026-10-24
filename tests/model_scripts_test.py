"""Offline regression tests for model preparation and exported-model parity.

All pixel arrays and ONNX models are generated here. These tests use no user
images, public model download, source checkpoint, or network connection.
"""

import importlib.util
import json
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
SPEC = importlib.util.spec_from_file_location("ovb_prepare", ROOT / "scripts/prepare_model.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class ModelPreparationContract(unittest.TestCase):
    def test_half_pixel_stretch_preserves_edges_and_fractional_center(self):
        pixels = np.array([0, 0, 0, 255, 255, 255, 255, 255], dtype=np.uint8)
        actual = PREPARE.preprocess_rgba(pixels, 2, 1, 3)
        self.assertEqual(actual.shape, (1, 3, 3, 3))
        self.assertEqual(actual.dtype, np.dtype("float32"))
        # Two black/white source pixels give exact 0, 127.5, 255 samples.
        # Each channel below is independently normalized by ImageNet mean/std.
        expected = np.array([
            [-2.11790393, 0.06550218, 2.24890830],
            [-2.03571429, 0.19642857, 2.42857143],
            [-1.80444444, 0.41777778, 2.64000000],
        ], dtype=np.float32)
        for row in range(3):
            np.testing.assert_allclose(actual[0, :, row, :], expected, rtol=0, atol=3e-7)

    def test_fixture_is_the_shared_rgba_recipe(self):
        image = PREPARE.fixture_rgba()
        self.assertEqual(image.shape, (32, 40, 4))
        self.assertEqual(image[0, 0].tolist(), [0, 0, 0, 255])
        self.assertEqual(image[31, 39].tolist(), [110, 163, 150, 255])

    def test_preprocessing_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            PREPARE.preprocess_rgba([0, 0, 0, 255], 2, 1, 3)
        with self.assertRaises(ValueError):
            PREPARE.preprocess_rgba([], 0, 1, 3)

    def test_corrupted_cached_checkpoint_fails_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="ovb-checkpoint-") as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"incorrect checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                PREPARE.download_public(PREPARE.SOURCE_URL, checkpoint, PREPARE.SOURCE_SHA256)

    def test_partial_alpha_is_composited_without_byte_rounding(self):
        actual = PREPARE.preprocess_rgba([1, 1, 1, 128], 1, 1, 1)
        # On white: 1*(128/255) + 255*(127/255), which is not an integer.
        pixel = 127 + 128 / 255
        expected = [(pixel / 255 - 0.485) / 0.229,
                    (pixel / 255 - 0.456) / 0.224,
                    (pixel / 255 - 0.406) / 0.225]
        np.testing.assert_allclose(actual[0, :, 0, 0], expected, atol=1e-7, rtol=0)


class ExportParityContract(unittest.TestCase):
    def run_case(self, expected, labels=None):
        with tempfile.TemporaryDirectory(prefix="ovb-parity-") as directory:
            root = Path(directory)
            graph = helper.make_graph([
                helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
                helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
                helper.make_node("MatMul", ["features", "weight"], ["projection"]),
                helper.make_node("Add", ["projection", "bias"], ["logits"]),
            ], "independent-parity-fixture",
                [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])],
                [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
                [numpy_helper.from_array(np.array([[1, -1], [2, -2], [3, -3]], dtype=np.float32), "weight"),
                 numpy_helper.from_array(np.array([0.5, -0.25], dtype=np.float32), "bias")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
            onnx.save(model, root / "model.onnx")
            (root / "report.json").write_text(json.dumps({
                "labels": labels if labels is not None else ["left", "right"],
                "parityFixture": {"input": [0.0] * (3 * 224 * 224), "shape": [1, 3, 224, 224], "expectedLogits": expected},
            }), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/predict.py"), "--model", str(root / "model.onnx"),
                "--verify-report", str(root / "report.json"),
            ], capture_output=True, text=True, check=False)
            return result.returncode, json.loads(result.stdout)

    def test_reads_and_executes_embedded_onnx_without_checkpoint(self):
        code, result = self.run_case([0.5, -0.25])
        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["logits"], [0.5, -0.25])

    def test_mismatched_browser_logits_fail_process(self):
        code, result = self.run_case([5.0, -0.25])
        self.assertNotEqual(code, 0)
        self.assertFalse(result["passed"])

    def test_nonfinite_browser_reference_is_rejected(self):
        code, result = self.run_case([float("nan"), -0.25])
        self.assertNotEqual(code, 0)
        self.assertFalse(result["passed"])

    def test_label_count_must_match_logits(self):
        code, result = self.run_case([0.5, -0.25], ["only"])
        self.assertNotEqual(code, 0)
        self.assertFalse(result["passed"])


class DeploymentKitContract(unittest.TestCase):
    def module(self):
        path = ROOT / "public/kit/predict.py"
        self.assertTrue(path.is_file(), "the portable inference example is missing")
        spec = importlib.util.spec_from_file_location("ovb_kit_predict", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_partial_alpha_is_composited_on_white_without_byte_rounding(self):
        module = self.module()
        with tempfile.TemporaryDirectory(prefix="ovb-kit-image-") as directory:
            path = Path(directory) / "partial.png"
            Image.new("RGBA", (1, 1), (1, 1, 1, 128)).save(path)
            actual = module.preprocess_image(path)
        pixel = 127 + 128 / 255
        expected = [(pixel / 255 - 0.485) / 0.229,
                    (pixel / 255 - 0.456) / 0.224,
                    (pixel / 255 - 0.406) / 0.225]
        np.testing.assert_allclose(actual[0, :, 0, 0], expected, atol=1e-7, rtol=0)

    def test_half_pixel_resize_uses_fractional_center(self):
        module = self.module()
        with tempfile.TemporaryDirectory(prefix="ovb-kit-image-") as directory:
            path = Path(directory) / "ramp.png"
            image = Image.new("RGBA", (2, 1))
            image.putdata([(0, 0, 0, 255), (255, 255, 255, 255)])
            image.save(path)
            actual = module.preprocess_image(path, size=3)
        self.assertEqual(actual.shape, (1, 3, 3, 3))
        np.testing.assert_allclose(actual[0, 0, 1], [-2.11790393, 0.06550218, 2.24890830], atol=3e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
