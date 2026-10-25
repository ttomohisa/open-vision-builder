"""Offline, hand-derived checks for the bounded classification comparison."""
import importlib.util
import copy
import hashlib
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
MODULE_PATH = ROOT / "scripts/classification_real_reference.py"


def subject():
    if not MODULE_PATH.is_file():
        raise AssertionError("The independent classification comparison is not implemented")
    spec = importlib.util.spec_from_file_location("classification_real_reference", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def models():
    image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])
    feature = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    frozen = [helper.make_node("Mul", ["images", "scale"], ["scaled"], name="scale"),
              helper.make_node("GlobalAveragePool", ["scaled"], ["pooled"], name="pool"),
              helper.make_node("Flatten", ["pooled"], ["features"], axis=1, name="flatten")]
    scale = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "scale")
    old = [numpy_helper.from_array(np.zeros((2, 3), dtype=np.float32), "old.weight"),
           numpy_helper.from_array(np.zeros(2, dtype=np.float32), "old.bias")]
    native = helper.make_model(helper.make_graph(
        frozen + [helper.make_node("Gemm", ["features", "old.weight", "old.bias"], ["logits"], name="old", transB=1)],
        "tiny-classifier", [image], [output, feature], [scale, *old]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    head = {"featureSize": 3, "classes": 2, "weights": [1., -1., 2., -2., 3., -3.], "bias": [.5, -.5]}
    new = [numpy_helper.from_array(np.asarray(head["weights"], dtype=np.float32).reshape(3, 2), "ovb.head.weight"),
           numpy_helper.from_array(np.asarray(head["bias"], dtype=np.float32), "ovb.head.bias")]
    exported = helper.make_model(helper.make_graph(
        frozen + [helper.make_node("MatMul", ["features", "ovb.head.weight"], ["ovb.head.matmul"], name="ovb.head.linear"),
                  helper.make_node("Add", ["ovb.head.matmul", "ovb.head.bias"], ["logits"], name="ovb.head.add")],
        "tiny-classifier", [image], [output], [scale, *new]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    helper.set_model_props(exported, {"ovb.labels": '["left","right"]', "ovb.recipe": "mobilenetv4-small"})
    return native, exported, head


class MetricsContract(unittest.TestCase):
    def test_confusion_macro_f1_and_cross_entropy_are_independently_derived(self):
        module = subject()
        # Probabilities [.75,.25] then [.25,.75], truths 0,0,1: one false negative.
        values = np.asarray([[math.log(3), 0], [0, math.log(3)], [0, math.log(3)]], dtype=np.float32)
        result = module.classification_metrics(values, [0, 0, 1], 2)
        self.assertEqual(result["confusion"], [[1, 1], [0, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macroF1"], 2 / 3)
        self.assertEqual(result["recall"], [.5, 1.])
        self.assertAlmostEqual(result["crossEntropy"], (-2 * math.log(.75) - math.log(.25)) / 3, places=7)

    def test_zero_denominators_and_tied_logits_have_deterministic_results(self):
        result = subject().classification_metrics(np.zeros((2, 3), dtype=np.float32), [0, 1], 3)
        self.assertEqual(result["predictions"], [0, 0])
        self.assertEqual(result["recall"], [1., 0., 0.])
        self.assertAlmostEqual(result["macroF1"], 2 / 9)

    def test_rejects_shape_label_and_nonfinite_data(self):
        module = subject()
        for values, labels, classes in [([[1., 2.]], [0, 1], 2), ([[math.nan, 1.]], [0], 2),
                                       ([[0., 1.]], [-1], 2), ([[0., 1.]], [True], 2)]:
            with self.subTest(values=values, labels=labels), self.assertRaises(ValueError):
                module.classification_metrics(values, labels, classes)

    def test_numeric_failure_is_reported_without_relaxing_tolerance(self):
        result = subject().compare_arrays([0., .001], [0., 0.], "one-image")
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatchedValues"], 1)
        self.assertAlmostEqual(result["maxToleranceRatio"], 5.)


class TrainingContract(unittest.TestCase):
    def test_zero_head_first_update_and_before_update_loss(self):
        result = subject().train_linear(np.asarray([[1., 2.], [3., 4.]], dtype=np.float32), [0, 1], 1, .1, 2)
        np.testing.assert_allclose(result["weights"], [[-.05, .05], [-.05, .05]], rtol=0, atol=1e-8)
        np.testing.assert_array_equal(result["bias"], [0., 0.])
        self.assertAlmostEqual(result["losses"][0], math.log(2), places=6)
        # Updated logits are [-.15,.15] and [-.35,.35], distinct from the saved initial loss.
        expected_final_ce = (math.log1p(math.exp(.3)) + math.log1p(math.exp(-.7))) / 2
        self.assertAlmostEqual(result["finalTrain"]["crossEntropy"], expected_final_ce, places=6)
        self.assertNotAlmostEqual(result["finalTrain"]["crossEntropy"], result["losses"][-1], places=5)


class BindingContract(unittest.TestCase):
    def plan(self):
        plan = json.loads((ROOT / "tests/fixtures/classification-real-plan.json").read_text())
        plan["datasetManifestSha256"] = "a" * 64
        return plan

    def test_recipe_and_criteria_cannot_be_changed_after_registration(self):
        module = subject()
        self.assertTrue(callable(getattr(module, "validate_plan", None)), "Plan validation is not implemented")
        module.validate_plan(self.plan(), "a" * 64)
        for section, key, value in [("training", "epochs", 41), ("criteria", "minimumTrainAccuracy", .8),
                                    ("tolerances", "atol", .01), ("budget", "automaticRetries", 1)]:
            plan = self.plan()
            plan[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                module.validate_plan(plan, "a" * 64)
        with self.assertRaises(ValueError):
            module.validate_plan(self.plan(), "b" * 64)

    def test_wrong_worker_split_or_encoded_hash_is_rejected(self):
        module = subject()
        self.assertTrue(callable(getattr(module, "validate_request", None)), "Actual Worker binding is not implemented")
        active = [{"id": "train", "label": 0, "split": "train", "pngSha256": "a" * 64},
                  {"id": "eval", "label": 1, "split": "evaluation", "pngSha256": "b" * 64}]
        request = {"labels": ["left", "right"], "epochs": 40, "learningRate": .01,
                   "samples": [{"id": "random1", "name": "train.png", "imageId": "train", "label": 0,
                                "split": "train", "fileSha256": "a" * 64},
                               {"id": "random2", "name": "eval.png", "imageId": "eval", "label": 1,
                                "split": "test", "fileSha256": "b" * 64}]}
        module.validate_request(request, active, ["left", "right"])
        for key, bad in [("label", 1), ("split", "test"), ("fileSha256", "b" * 64)]:
            changed = copy.deepcopy(request)
            changed["samples"][0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_request(changed, active, ["left", "right"])

    def test_prepared_artifact_pin_is_recorded_in_plan_not_historical_bytes(self):
        plan = self.plan()
        plan["model"]["onnxSha256"] = "f" * 64
        subject().validate_plan(plan, "a" * 64)

    def test_manifest_order_and_tuning_exclusion_are_enforced(self):
        module = subject()
        self.assertTrue(callable(getattr(module, "validate_manifest", None)), "Manifest validation is not implemented")
        plan = self.plan()
        images = []
        for label in range(5):
            for index, split in enumerate(["train"] * 6 + ["tuning"] * 2 + ["evaluation"] * 4):
                identity = f"image-{label}-{index}"
                images.append({"id": identity, "label": label, "split": split,
                               "creatorKey": identity, "flickrId": identity,
                               "originalSha256": f"{label * 12 + index:064x}",
                               "pngSha256": f"{100 + label * 12 + index:064x}"})
        manifest = {"schemaVersion": 1, "classes": plan["classes"], "images": images}
        actual = module.validate_manifest(manifest, plan)
        self.assertEqual(len(actual), 50)
        self.assertTrue(all(item["split"] != "tuning" for item in actual))
        changed = copy.deepcopy(manifest)
        changed["images"][1]["creatorKey"] = changed["images"][0]["creatorKey"]
        with self.assertRaises(ValueError):
            module.validate_manifest(changed, plan)


class IndependentInputContract(unittest.TestCase):
    def test_bad_public_checkpoint_is_rejected_before_native_model_loading(self):
        module = subject()
        self.assertTrue(callable(getattr(module, "load_native", None)), "Pinned native loader is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors").write_bytes(b"not-the-pinned-checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                module.load_native(Path(directory))

    def test_independent_png_preprocessing_rejects_changed_browser_tensor(self):
        module = subject()
        self.assertTrue(callable(getattr(module, "read_image_input", None)), "Input comparison is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(root / "white.png")
            png = (root / "white.png").read_bytes()
            item = {"id": "white", "width": 1, "height": 1, "pngBytes": len(png), "pngSha256": hashlib.sha256(png).hexdigest()}
            values = np.empty((1, 3, 224, 224), dtype="<f4")
            values[:, 0] = 2.2489083
            values[:, 1] = 2.4285715
            values[:, 2] = 2.64
            payload = values.tobytes()
            (root / "inputs/white.f32").write_bytes(payload)
            row = {"inputSha256": hashlib.sha256(payload).hexdigest()}
            actual, result = module.read_image_input(root, root, item, row)
            self.assertTrue(result["passed"])
            self.assertEqual(actual.shape, (1, 3, 224, 224))
            values[0, 0, 0, 0] += .001
            payload = values.tobytes()
            (root / "inputs/white.f32").write_bytes(payload)
            row["inputSha256"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                module.read_image_input(root, root, item, row)


class CompletionContract(unittest.TestCase):
    def test_actual_worker_report_is_bound_to_the_same_update_count_and_model(self):
        module = subject()
        plan = json.loads((ROOT / "tests/fixtures/classification-real-plan.json").read_text())
        payload = b"encoded-model-byte-identity"
        digest = hashlib.sha256(payload).hexdigest()
        bindings = {"planSha256": "a" * 64, "datasetManifestSha256": "b" * 64}
        active, samples, rows, predictions = [], [], [], []
        for label in range(5):
            for index in range(10):
                image_id, worker_id = f"image-{label}-{index}", f"worker-{label}-{index}"
                split = "train" if index < 6 else "evaluation"
                active.append({"id": image_id, "label": label, "split": split, "pngSha256": "c" * 64})
                samples.append({"id": worker_id, "imageId": image_id, "name": image_id + ".png", "label": label,
                                "split": "train" if index < 6 else "test", "fileSha256": "c" * 64})
                rows.append({"id": image_id, "label": label, "split": split, "features": [0.] * 1280,
                             "baseLogits": [0.] * 1000, "finalLogits": [0.] * 5})
                if index >= 6:
                    predictions.append({"id": worker_id, "actual": label, "predicted": 0, "scores": [.2] * 5})
        losses = [math.log(5)] * 40
        summary = {"epochs": 40, "trainingCount": 30, "testCount": 20, "backend": "wasm", "accuracy": .2,
                   "modelSha256": digest, "modelBytes": len(payload), "losses": losses, "predictions": predictions,
                   "durationMs": 1., "nativeMaxAbsError": 0., "exportMaxAbsError": 0., "weightDeltaL2": .08}
        browser = {"schemaVersion": 1, **bindings, "modelSha256": digest, "sourceModelSha256": plan["model"]["onnxSha256"],
                   "exportedModelBytes": len(payload), "labels": plan["classes"], "summary": summary,
                   "request": {"labels": plan["classes"], "epochs": 40, "learningRate": .01, "samples": samples},
                   "head": {"classes": 5, "featureSize": 1280, "weights": [.001] * 6400, "bias": [0.] * 5}, "images": rows}
        fresh = {"schemaVersion": 1, **bindings, "modelSha256": digest,
                 "images": [{"id": row["id"], "logits": [0.] * 5} for row in rows]}
        report = {"schemaVersion": 1, "labels": plan["classes"], "artifact": {"file": "model.onnx", "sha256": digest, "bytes": len(payload)},
                  "model": {"sha256": module.SOURCE_SHA256, "revision": module.SOURCE_REVISION, "modelId": module.MODEL_ID},
                  "training": {"scope": "classifier-head-only", "epochsCompleted": 40, "learningRate": .01, "backend": "wasm",
                               "engine": "@jax-js/jax@0.1.27", "lossTiming": "before-each-update", "losses": losses, "weightDeltaL2": .08},
                  "verification": {"atol": .0002, "rtol": .0001, "native": {"passed": True}, "export": {"passed": True, "images": 21}},
                  "metrics": {"accuracy": .2, "trainingCount": 30, "testCount": 20, "durationMs": 1.}}
        self.assertTrue(all(check["passed"] for check in module.validate_records(plan, active, browser, fresh, report, bindings, payload)))
        wrong_timing = copy.deepcopy(report)
        wrong_timing["training"]["lossTiming"] = "after-each-update"
        with self.assertRaisesRegex(ValueError, "loss timing"):
            module.validate_records(plan, active, browser, fresh, wrong_timing, bindings, payload)
        wrong_artifact = copy.deepcopy(fresh)
        wrong_artifact["modelSha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "model digest"):
            module.validate_records(plan, active, browser, wrong_artifact, report, bindings, payload)

    def test_exact_five_point_gap_passes_but_a_larger_gap_fails(self):
        strong = {"initialTrain": {"crossEntropy": 1.}, "finalTrain": {"crossEntropy": .5, "accuracy": 1.},
                  "evaluation": {"accuracy": .8, "macroF1": .8}}
        boundary = copy.deepcopy(strong)
        boundary["evaluation"] = {"accuracy": .75, "macroF1": .75}
        checks = subject().quality_checks(strong, boundary)
        self.assertTrue(all(item["passed"] for item in checks))
        outside = copy.deepcopy(boundary)
        outside["evaluation"] = {"accuracy": .749999, "macroF1": .749999}
        failures = [item["name"] for item in subject().quality_checks(strong, outside) if not item["passed"]]
        self.assertEqual(failures, ["evaluation-accuracy-gap", "evaluation-macroF1-gap"])

    def test_quality_failure_remains_explicit_in_completed_result(self):
        module = subject()
        self.assertTrue(callable(getattr(module, "quality_checks", None)), "Quality conditions are not implemented")
        weak = {"initialTrain": {"crossEntropy": 1.}, "finalTrain": {"crossEntropy": .9, "accuracy": .5},
                "evaluation": {"accuracy": .5, "macroF1": .5}}
        conditions = module.quality_checks(weak, weak)
        self.assertFalse(all(item["passed"] for item in conditions))
        self.assertEqual(len([item for item in conditions if not item["passed"]]), 6)

    def test_cli_integrity_error_is_nonzero_and_saves_failed_report(self):
        self.assertTrue(MODULE_PATH.is_file())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, str(MODULE_PATH), "--plan", str(root / "absent.json"),
                                     "--data", str(root / "data.json"), "--images", str(root), "--run", str(root),
                                     "--source-model", str(root / "source.onnx"), "--source-cache", str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, "Missing prerequisites must not be reported as a completed experiment")
            report = json.loads((root / "comparison.json").read_text())
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["passed"])


class ExportContract(unittest.TestCase):
    def test_keeps_frozen_ancestors_and_replaces_entire_classifier(self):
        native, exported, head = models()
        result = subject().validate_export(native.SerializeToString(), exported.SerializeToString(), head, ["left", "right"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["frozenInitializers"], 1)

    def test_rejects_changed_frozen_weight(self):
        native, exported, head = models()
        exported.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.asarray([2.], dtype=np.float32), "scale"))
        with self.assertRaisesRegex(ValueError, "frozen"):
            subject().validate_export(native.SerializeToString(), exported.SerializeToString(), head, ["left", "right"])

    def test_rejects_changed_frozen_graph(self):
        native, exported, head = models()
        exported.graph.node[0].op_type = "Add"
        with self.assertRaisesRegex(ValueError, "frozen"):
            subject().validate_export(native.SerializeToString(), exported.SerializeToString(), head, ["left", "right"])

    def test_rejects_changed_head_even_if_predictions_could_be_close(self):
        native, exported, head = models()
        head["weights"][0] += .00001
        with self.assertRaisesRegex(ValueError, "head"):
            subject().validate_export(native.SerializeToString(), exported.SerializeToString(), head, ["left", "right"])

    def test_rejects_external_weights_without_following_paths(self):
        native, exported, head = models()
        tensor = exported.graph.initializer[0]
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="must-not-be-read")
        with self.assertRaisesRegex(ValueError, "external"):
            subject().validate_export(native.SerializeToString(), exported.SerializeToString(), head, ["left", "right"])


if __name__ == "__main__":
    unittest.main()
