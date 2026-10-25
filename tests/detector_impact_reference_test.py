"""Offline contracts for the bounded impact experiment; no pretrained inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ImpactReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "scripts/detector_impact_reference.py"
        if not path.is_file():
            return
        spec = importlib.util.spec_from_file_location("impact_reference", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(hasattr(self, "module"), "Missing bounded impact reference implementation")

    def test_tensor_contract_rejects_reshaping_nonfinite_boolean_and_missing_values(self):
        valid = {"shape": [2, 2], "data": [1, 2, 3, 4]}
        actual = self.module.checked_tensor(valid, [2, 2], "fixture")
        np.testing.assert_array_equal(actual, [[1, 2], [3, 4]])
        for bad in [
            {"shape": [4], "data": [1, 2, 3, 4]},
            {"shape": [2, 2], "data": [1, 2, 3]},
            {"shape": [2, 2], "data": [1, 2, 3, float("nan")]},
            {"shape": [2, 2], "data": [1, 2, 3, 1e100]},
            {"shape": [2, 2], "data": [1, 2, 3, True]},
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.module.checked_tensor(bad, [2, 2], "fixture")
        with self.assertRaises(ValueError):
            self.module.checked_tensor({"shape": [True, 2], "data": [1, 2]}, [1, 2], "fixture")

    def test_canonical_input_hash_length_shape_and_path_are_checked(self):
        values = np.zeros((1, 3, 640, 640), dtype="<f4")
        payload = values.tobytes()
        item = {"id": "000000000001", "input": {
            "file": "inputs/000000000001.f32", "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "shape": [1, 3, 640, 640],
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            (root / item["input"]["file"]).write_bytes(payload)
            actual = self.module.read_input(root, item)
            self.assertEqual(actual.shape, (1, 3, 640, 640))
            for field, changed in [("sha256", "0" * 64), ("bytes", 1),
                                   ("shape", [1, 3, 1280, 320]), ("file", "../outside.f32")]:
                bad = copy.deepcopy(item)
                bad["input"][field] = changed
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.module.read_input(root, bad)

    def test_preregistration_without_bound_manifest_fails_before_loading_inputs(self):
        plan = json.loads((ROOT / "tests/fixtures/detector-impact-plan.json").read_text())
        plan["datasetManifestSha256"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "data.json").write_text(json.dumps({"schemaVersion": 1, "images": []}))
            with self.assertRaisesRegex(ValueError, "manifest|preregister"):
                self.module.load_experiment(root / "plan.json", root / "data.json", root)

    def test_changed_decision_conditions_are_rejected_before_measurement(self):
        plan = json.loads((ROOT / "tests/fixtures/detector-impact-plan.json").read_text())
        for key in ("minimumTrainLossReduction", "maximumOwnLossRelativeGap", "maximumEvaluationAp50Gap", "maximumEvaluationAp50Drop"):
            changed = copy.deepcopy(plan)
            changed["criteria"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "criteria"):
                self.module.validate_plan(changed, plan["datasetManifestSha256"])

    def test_duplicate_detection_cannot_match_an_object_twice(self):
        images = [{"id": "a", "targets": {"labels": [0, 0], "boxes": [.2, .2, .1, .1, .8, .8, .1, .1]}}]
        detections = {"a": [
            {"classIndex": 0, "queryIndex": 0, "score": .9, "box": [.2, .2, .1, .1]},
            {"classIndex": 0, "queryIndex": 1, "score": .8, "box": [.2, .2, .1, .1]},
            {"classIndex": 0, "queryIndex": 2, "score": .7, "box": [.8, .8, .1, .1]},
        ]}
        result = self.module.detection_metrics(images, detections)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 0))
        self.assertAlmostEqual(result["ap50"], 0.834983498349835, places=12)

    def test_wrong_class_and_absent_class_predictions_are_false_positives(self):
        images = [{"id": "a", "targets": {"labels": [0], "boxes": [.5, .5, .2, .2]}}]
        wrong = {"a": [{"classIndex": 79, "queryIndex": 0, "score": .9, "box": [.5, .5, .2, .2]}]}
        result = self.module.detection_metrics(images, wrong)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["ap50"]), (0, 1, 1, 0))
        self.assertEqual(result["classesWithGroundTruth"], [0])

    def test_missing_class_reduces_mean_ap_and_empty_predictions_keep_false_negatives(self):
        images = [{"id": "a", "targets": {"labels": [0, 1], "boxes": [.2, .2, .1, .1, .8, .8, .1, .1]}}]
        detections = {"a": [{"classIndex": 0, "queryIndex": 0, "score": .9, "box": [.2, .2, .1, .1]}]}
        result = self.module.detection_metrics(images, detections)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["ap50"]), (1, 0, 1, .5))
        empty = self.module.detection_metrics(images, {"a": []})
        self.assertEqual((empty["tp"], empty["fp"], empty["fn"], empty["ap50"]), (0, 0, 2, 0))

    def test_global_top_selection_uses_flat_query_class_ties_and_inclusive_threshold(self):
        logits = np.full((300, 80), -100, dtype=np.float32)
        logits[0, 3] = logits[1, 0] = 0
        boxes = np.tile([.5, .5, .2, .2], (300, 1))
        prediction = {"logits": {"shape": [300, 80], "data": logits.ravel().tolist()},
                      "boxes": {"shape": [300, 4], "data": boxes.ravel().tolist()}}
        result = self.module.select_detections(prediction, score_threshold=.5, max_detections=1)
        self.assertEqual([(item["queryIndex"], item["classIndex"], item["score"]) for item in result], [(0, 3, .5)])

    def measurement(self):
        trainable = [f"model.decoder.bbox_embed.2.layers.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")]
        trainable += [f"model.decoder.class_embed.2.{kind}" for kind in ("weight", "bias")]
        frozen = [f"model.decoder.lqe_layers.2.reg_conf.layers.{i}.{kind}" for i in range(2) for kind in ("weight", "bias")]
        tensor = lambda value: {"shape": [1], "data": [value]}
        return {
            "prediction": {name: tensor(.5) for name in ("logits", "boxes", "corners")},
            "losses": {name: 1.0 for name in ("vfl", "l1", "giou", "total")},
            "matches": {"predictionIndices": [2], "targetIndices": [0]},
            "gradients": {name: tensor(.1) for name in trainable},
            "nextWeights": {name: tensor(.2) for name in trainable + frozen},
        }

    def test_control_detects_gradient_update_matching_and_frozen_changes(self):
        limits = json.loads((ROOT / "tests/fixtures/detector-training-tolerances.json").read_text())
        expected = self.measurement()
        good = self.module.compare_measurements(copy.deepcopy(expected), expected, limits)
        self.assertTrue(good["passed"])
        changes = [
            lambda actual: actual["gradients"]["model.decoder.class_embed.2.bias"]["data"].__setitem__(0, 3),
            lambda actual: actual["nextWeights"]["model.decoder.class_embed.2.bias"]["data"].__setitem__(0, 3),
            lambda actual: actual["matches"]["predictionIndices"].__setitem__(0, 3),
            lambda actual: actual["nextWeights"]["model.decoder.lqe_layers.2.reg_conf.layers.1.bias"]["data"].__setitem__(0, .2000001),
        ]
        for change in changes:
            actual = copy.deepcopy(expected)
            change(actual)
            self.assertFalse(self.module.compare_measurements(actual, expected, limits)["passed"])
        self.assertEqual(expected, self.measurement(), "Comparing a control must not mutate its reference")

    def test_control_rejects_missing_parameter_even_when_remaining_values_match(self):
        limits = json.loads((ROOT / "tests/fixtures/detector-training-tolerances.json").read_text())
        expected = self.measurement()
        actual = copy.deepcopy(expected)
        del actual["gradients"]["model.decoder.class_embed.2.bias"]
        with self.assertRaisesRegex(ValueError, "gradient|parameter|keys"):
            self.module.compare_measurements(actual, expected, limits)

    def traces(self):
        plan = json.loads((ROOT / "tests/fixtures/detector-impact-plan.json").read_text())
        plan["datasetManifestSha256"] = "b" * 64
        logits = np.full((300, 80), -100, dtype=np.float32)
        logits[0, 0] = 2
        prediction = {"logits": {"shape": [300, 80], "data": logits.ravel().tolist()},
                      "boxes": {"shape": [300, 4], "data": np.tile([.5, .5, .2, .2], (300, 1)).ravel().tolist()},
                      "corners": {"shape": [300, 132], "data": [0.0] * (300 * 132)}}
        images = [{"id": f"{index + 1:012d}", "split": "train" if index < 4 else "tuning" if index < 6 else "evaluation",
                   "targets": {"labels": [0], "boxes": [.5, .5, .2, .2]}, "inputSha256": "c" * 64}
                  for index in range(10)]
        sample = self.measurement()
        initial_weights = copy.deepcopy(sample["nextWeights"])
        for key in sample["gradients"]:
            sample["nextWeights"][key]["data"][0] = .3
        trace = {"schemaVersion": 1, "planSha256": "a" * 64, "datasetManifestSha256": "b" * 64,
                 "images": images, "frozenUnchanged": True,
                 "initial": [{"id": item["id"], "prediction": prediction, "matches": sample["matches"],
                              "losses": {name: 2.0 for name in ("vfl", "l1", "giou", "total")}} for item in images],
                 "final": [{"id": item["id"], "prediction": prediction, "matches": sample["matches"],
                            "losses": {name: 1.0 for name in ("vfl", "l1", "giou", "total")}} for item in images],
                 "steps": [{"id": images[index % 4]["id"], "step": index + 1, **copy.deepcopy(sample), "prediction": prediction}
                           for index in range(12)],
                 "checkpoints": {"0": initial_weights, "6": copy.deepcopy(sample["nextWeights"]), "12": copy.deepcopy(sample["nextWeights"])}}
        return copy.deepcopy(trace), trace, plan

    def test_summary_rejects_missing_steps_and_preserves_original_qualification(self):
        browser, native, plan = self.traces()
        result = self.module.summarize_comparison(browser, native, plan)
        self.assertTrue(result["passed"])
        self.assertEqual(result["arms"]["browser"]["trainLossReduction"], .5)
        self.assertEqual(result["evaluationAp50Gap"], 0)
        self.assertEqual(result["detectorQualification"], "FAIL")
        self.assertFalse(result["releaseReady"])
        browser["steps"].pop(3)
        with self.assertRaisesRegex(ValueError, "steps"):
            self.module.summarize_comparison(browser, native, plan)

    def test_summary_independently_rejects_frozen_changes_and_checkpoint_substitution(self):
        browser, native, plan = self.traces()
        browser["steps"][0]["nextWeights"]["model.decoder.lqe_layers.2.reg_conf.layers.1.bias"]["data"][0] = .21
        with self.assertRaisesRegex(ValueError, "frozen|LQE"):
            self.module.summarize_comparison(browser, native, plan)
        browser, native, plan = self.traces()
        browser["checkpoints"]["6"]["model.decoder.class_embed.2.bias"]["data"][0] = .4
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.module.summarize_comparison(browser, native, plan)


if __name__ == "__main__":
    unittest.main()
