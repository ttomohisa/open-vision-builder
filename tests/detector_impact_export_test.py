"""Offline export verification using a tiny real graph, never the pretrained model."""
import copy
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_detector_impact_export.py"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def tiny_model(bias):
    graph = helper.make_graph([
        helper.make_node("Flatten", ["images"], ["flattened"], axis=1),
        helper.make_node("Gather", ["flattened", "index"], ["sample"], axis=1),
        helper.make_node("Add", ["sample", "head_bias"], ["shifted"]),
        helper.make_node("Add", ["zero_logits", "shifted"], ["logits"]),
        helper.make_node("Identity", ["box_values"], ["boxes"]),
    ], "impact-export-fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 300, 80]),
         helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 300, 4])],
        [numpy_helper.from_array(np.array([0], dtype=np.int64), "index"),
         numpy_helper.from_array(np.array([bias], dtype=np.float32), "head_bias"),
         numpy_helper.from_array(np.zeros((1, 300, 80), dtype=np.float32), "zero_logits"),
         numpy_helper.from_array(np.tile(np.array([.5, .5, .25, .125], dtype=np.float32),
                                       (1, 300, 1)), "box_values")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def fixture(root):
    """Hand-derived outputs are the selected input pixel plus the trained .25 bias."""
    (root / "inputs").mkdir()
    manifest = {"schemaVersion": 1, "images": []}
    predictions = []
    for image_id, pixel, split in [("a", .25, "train"), ("b", .5, "evaluation")]:
        values = np.zeros((1, 3, 640, 640), dtype="<f4")
        values[0, 0, 0, 0] = pixel
        payload = values.tobytes()
        relative = f"inputs/{image_id}.f32"
        (root / relative).write_bytes(payload)
        manifest["images"].append({"id": image_id, "split": split, "input": {
            "file": relative, "sha256": digest(payload), "bytes": len(payload), "shape": [1, 3, 640, 640]},
            "targets": {"labels": [0], "boxes": [.5, .5, .25, .125]}})
        predictions.append({"id": image_id, "inputSha256": digest(payload), "prediction": {
            "logits": {"shape": [300, 80], "data": [pixel + .25] * 24000},
            "boxes": {"shape": [300, 4], "data": [.5, .5, .25, .125] * 300}}})
    source, model = tiny_model(0), tiny_model(.25)
    recipe = {"schemaVersion": 1, "source": {"sha256": "a" * 64}, "artifacts": {
        "detector.onnx": {"sha256": digest(source.SerializeToString()), "bytes": len(source.SerializeToString())}},
        "exportRecipe": {"schemaVersion": 1, "id": "dfine-nano-coco-final-head-v1",
            "input": {"name": "images", "shape": [1, 3, 640, 640]},
            "outputs": [{"name": "logits", "shape": [1, 300, 80]}, {"name": "boxes", "shape": [1, 300, 4]}],
            "parameters": [{"parameter": "head.bias", "initializers": ["head_bias"], "shape": [1]}]}}
    plan = {"schemaVersion": 1, "id": "dfine-coco-own-prefix-impact-v1", "sourceCheckpointSha256": "a" * 64,
            "criteria": {"exportLogits": {"atol": .0002, "rtol": .0001},
                         "exportBoxes": {"atol": .00002, "rtol": .0001}}}
    browser = {"schemaVersion": 1, "final": copy.deepcopy(predictions),
               "checkpoints": {"12": {"head.bias": {"shape": [1], "data": [.25]}}}}
    fresh = {"schemaVersion": 1, "images": copy.deepcopy(predictions)}
    return {"manifest": manifest, "plan": plan, "recipe": recipe, "browser": browser,
            "fresh": fresh, "source": source, "model": model}


def write_fixture(root, data, bind=True):
    for name in ("source", "model"):
        (root / f"{name}.onnx").write_bytes(data[name].SerializeToString())
    (root / "manifest.json").write_text(json.dumps(data["manifest"]), encoding="utf-8")
    if bind:
        data["plan"]["datasetManifestSha256"] = digest((root / "manifest.json").read_bytes())
    (root / "plan.json").write_text(json.dumps(data["plan"]), encoding="utf-8")
    if bind:
        data["browser"]["datasetManifestSha256"] = data["plan"]["datasetManifestSha256"]
        data["browser"]["planSha256"] = digest((root / "plan.json").read_bytes())
        payload = (root / "model.onnx").read_bytes()
        data["browser"]["artifact"] = {"sha256": digest(payload), "bytes": len(payload)}
        data["fresh"]["modelSha256"] = digest(payload)
    for name in ("recipe", "browser", "fresh"):
        (root / f"{name}.json").write_text(json.dumps(data[name]), encoding="utf-8")


class ImpactExportVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.is_file():
            return
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("impact_export", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(hasattr(self, "module"), "Missing impact export verifier implementation")

    def verify(self, root):
        return self.module.verify(root / "model.onnx", root / "source.onnx", root / "manifest.json",
                                  root / "plan.json", root, root / "browser.json", root / "fresh.json",
                                  root / "recipe.json")

    def test_complete_export_runs_all_inputs_and_compares_three_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            write_fixture(root, data)
            result = self.verify(root)
        self.assertTrue(result["passed"])
        self.assertFalse(result["releaseReady"])
        self.assertEqual(result["detectorQualification"], "FAIL")
        self.assertEqual(result["frozenInitializerCount"], 3)
        self.assertEqual(result["onnxruntimeVersion"], "1.20.1")
        self.assertEqual([item["id"] for item in result["images"]], ["a", "b"])
        for item in result["images"]:
            self.assertEqual(set(item["comparison"]), {"cpuVsBrowserFinal", "cpuVsFreshBrowser", "freshBrowserVsBrowserFinal"})
            for pair in item["comparison"].values():
                for metric in pair.values():
                    self.assertEqual(metric["maxAbsoluteError"], 0)
                    self.assertTrue(metric["passed"])

    def test_original_tolerances_reject_small_final_and_fresh_output_errors(self):
        for trace, name, delta in [("browser", "logits", .0004), ("fresh", "boxes", .0001)]:
            with self.subTest(trace=trace), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = fixture(root)
                rows = data[trace]["final" if trace == "browser" else "images"]
                rows[1]["prediction"][name]["data"][0] += delta
                write_fixture(root, data)
                result = self.verify(root)
                self.assertFalse(result["passed"])
                self.assertTrue(result["images"][0]["passed"])
                pair = "cpuVsBrowserFinal" if trace == "browser" else "cpuVsFreshBrowser"
                self.assertEqual(result["images"][1]["comparison"][pair][name]["mismatchedValues"], 1)
                self.assertFalse(result["images"][1]["comparison"]["freshBrowserVsBrowserFinal"][name]["passed"])

    def test_graph_frozen_initializer_mapped_weight_and_contract_changes_are_rejected(self):
        def frozen(model):
            model.graph.initializer[2].raw_data = np.ones((1, 300, 80), dtype=np.float32).tobytes()
        def mapped(model):
            model.graph.initializer[1].raw_data = np.array([.3], dtype=np.float32).tobytes()
        def graph(model):
            model.graph.node[2].op_type = "Sub"
        def shape(model):
            model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 320
        def duplicate(model):
            model.graph.initializer.append(model.graph.initializer[0])
        def external(model):
            tensor = TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
            tensor.external_data.add(key="location", value="must-not-read.bin")
            model.graph.node.append(helper.make_node("Constant", [], ["unused"], value=tensor))
        for change in [frozen, mapped, graph, shape, duplicate, external]:
            with self.subTest(change=change.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = fixture(root)
                change(data["model"])
                write_fixture(root, data)
                with self.assertRaises(ValueError):
                    self.verify(root)

    def test_missing_duplicate_reshaped_nonfinite_and_boolean_predictions_are_rejected(self):
        changes = [
            lambda data: data["fresh"]["images"].pop(),
            lambda data: data["fresh"]["images"][1].__setitem__("id", "a"),
            lambda data: data["browser"]["final"][0]["prediction"]["logits"].__setitem__("shape", [1, 300, 80]),
            lambda data: data["fresh"]["images"][0]["prediction"]["boxes"]["data"].__setitem__(0, float("nan")),
            lambda data: data["browser"]["final"][0]["prediction"]["logits"]["data"].__setitem__(0, True),
            lambda data: data["browser"]["checkpoints"]["12"]["head.bias"].__setitem__("shape", [1, 1]),
            lambda data: data["browser"]["checkpoints"]["12"]["head.bias"]["data"].__setitem__(0, 1e100),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = fixture(root)
                change(data)
                write_fixture(root, data)
                with self.assertRaises(ValueError):
                    self.verify(root)

    def test_every_artifact_digest_and_preregistered_tolerance_is_bound(self):
        changes = [
            lambda data: data["plan"].__setitem__("datasetManifestSha256", None),
            lambda data: data["browser"].__setitem__("datasetManifestSha256", "0" * 64),
            lambda data: data["browser"].__setitem__("planSha256", "0" * 64),
            lambda data: data["browser"]["artifact"].__setitem__("sha256", "0" * 64),
            lambda data: data["browser"]["artifact"].__setitem__("bytes", 1),
            lambda data: data["fresh"].__setitem__("modelSha256", "0" * 64),
            lambda data: data["fresh"]["images"][0].__setitem__("inputSha256", "0" * 64),
            lambda data: data["recipe"]["artifacts"]["detector.onnx"].__setitem__("sha256", "0" * 64),
            lambda data: data["plan"]["criteria"]["exportLogits"].__setitem__("atol", 1),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = fixture(root)
                write_fixture(root, data)
                change(data)
                write_fixture(root, data, bind=False)
                with self.assertRaises(ValueError):
                    self.verify(root)

    def test_canonical_input_shape_bytes_hash_path_and_finite_values_are_checked(self):
        for field, bad in [("file", "../outside.f32"), ("file", "/tmp/outside.f32"),
                           ("shape", [1, 3, 1280, 320]), ("bytes", 4), ("sha256", "0" * 64)]:
            with self.subTest(field=field, bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = fixture(root)
                data["manifest"]["images"][0]["input"][field] = bad
                write_fixture(root, data)
                with self.assertRaises(ValueError):
                    self.verify(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            payload = np.full((1, 3, 640, 640), float("nan"), dtype="<f4").tobytes()
            (root / "inputs/a.f32").write_bytes(payload)
            data["manifest"]["images"][0]["input"]["sha256"] = digest(payload)
            data["fresh"]["images"][0]["inputSha256"] = digest(payload)
            write_fixture(root, data)
            with self.assertRaises(ValueError):
                self.verify(root)

    def test_cli_replaces_stale_success_with_atomic_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            write_fixture(root, data)
            (root / "model.onnx").write_bytes(b"corrupt")
            output = root / "export-verification.json"
            output.write_text('{"passed":true}')
            arguments = [sys.executable, str(SCRIPT)]
            for option, value in [("model", "model.onnx"), ("source", "source.onnx"),
                                  ("manifest", "manifest.json"), ("plan", "plan.json"),
                                  ("browser", "browser.json"), ("fresh-browser", "fresh.json"),
                                  ("recipe", "recipe.json"), ("output", "export-verification.json")]:
                arguments.extend([f"--{option}", str(root / value)])
            arguments.extend(["--data-dir", str(root)])
            run = subprocess.run(arguments, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(run.stderr, "")
            report = json.loads(output.read_text())
            self.assertEqual(report, json.loads(run.stdout))
            self.assertFalse(report["passed"])
            self.assertFalse(report["releaseReady"])
            self.assertEqual(report["detectorQualification"], "FAIL")
            self.assertIn("error", report)
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
