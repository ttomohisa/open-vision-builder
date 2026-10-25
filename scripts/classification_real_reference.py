#!/usr/bin/env python3
"""Independent, bounded native classification reference and export verification.

Only fixed public images are accepted. Scientific failures remain explicit in
comparison.json; malformed or unbound evidence is an execution error. The native
arm computes its own timm features and never trains on browser feature tensors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
ATOL = 2e-4
RTOL = 1e-4
INPUT_SHAPE = [1, 3, 224, 224]
SOURCE_SHA256 = "7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4"
SOURCE_REVISION = "331fb803779522b685cf942e15f914fb6741c1eb"
MODEL_ID = "timm/mobilenetv4_conv_small.e2400_r224_in1k"


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    payload = Path(path).read_bytes()
    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, sha256(payload)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Refusing a symlink report")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite_array(value, shape=None, name="tensor", dtype=np.float64):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.size == 0:
        raise ValueError(f"{name} must contain numeric nonempty data")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(dtype)
    if (shape is not None and list(result.shape) != list(shape)) or not np.isfinite(result).all():
        raise ValueError(f"{name} shape or finite values invalid; expected {shape}")
    return result


def checked_labels(labels, count, classes):
    raw = np.asarray(labels)
    if (raw.dtype.kind not in "iu" or raw.shape != (count,) or np.any(raw < 0)
            or np.any(raw >= classes) or any(isinstance(value, (bool, np.bool_)) for value in labels)):
        raise ValueError("Invalid labels, class range, or row count")
    return raw.astype(np.int64)


def compare_arrays(actual, expected, name):
    expected = finite_array(expected, name=f"{name} expected")
    actual = finite_array(actual, expected.shape, f"{name} actual")
    error = np.abs(actual - expected)
    allowance = ATOL + RTOL * np.abs(expected)
    return {"name": name, "passed": bool(np.all(error <= allowance)),
            "maxAbsoluteError": float(error.max()), "maxToleranceRatio": float((error / allowance).max()),
            "mismatchedValues": int(np.count_nonzero(error > allowance)), "values": int(error.size)}


def classification_metrics(logits, labels, classes):
    values = finite_array(logits, name="classification logits")
    if type(classes) is not int or classes < 2 or values.ndim != 2 or values.shape[1] != classes:
        raise ValueError("Classification logits must have shape [images, classes]")
    labels = checked_labels(labels, values.shape[0], classes)
    shifted = values - values.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    predictions = values.argmax(axis=1)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    correct = np.diag(confusion).astype(np.float64)
    actual_count, predicted_count = confusion.sum(axis=1), confusion.sum(axis=0)
    recall = np.divide(correct, actual_count, out=np.zeros(classes), where=actual_count != 0)
    precision = np.divide(correct, predicted_count, out=np.zeros(classes), where=predicted_count != 0)
    denominator = actual_count + predicted_count
    f1 = np.divide(2 * correct, denominator, out=np.zeros(classes), where=denominator != 0)
    return {"count": len(labels), "accuracy": float(np.mean(predictions == labels)),
            "crossEntropy": float(-log_probabilities[np.arange(len(labels)), labels].mean()),
            "macroF1": float(f1.mean()), "confusion": confusion.tolist(), "recall": recall.tolist(),
            "precision": precision.tolist(), "classF1": f1.tolist(), "predictions": predictions.tolist(),
            "probabilities": np.exp(log_probabilities).tolist()}


def train_linear(features, labels, epochs, learning_rate, classes):
    import torch
    values = finite_array(features, name="native training features", dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1 or type(epochs) is not int or epochs < 1:
        raise ValueError("Invalid training feature shape or epoch count")
    if type(classes) is not int or classes < 2 or isinstance(learning_rate, bool) or not 0 < learning_rate <= 1:
        raise ValueError("Invalid class count or learning rate")
    targets = checked_labels(labels, len(values), classes)
    if set(targets.tolist()) != set(range(classes)):
        raise ValueError("Every training class must be represented")
    torch.set_num_threads(1)
    x, y = torch.from_numpy(values.copy()), torch.from_numpy(targets)
    weights = torch.nn.Parameter(torch.zeros((values.shape[1], classes), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(classes, dtype=torch.float32))
    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate, momentum=0, weight_decay=0, foreach=False)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(x @ weights + bias, y, reduction="mean")
        if not torch.isfinite(loss):
            raise ValueError("Native training loss is nonfinite")
        losses.append(float(loss.detach()))  # Before this update, matching trainHead.
        loss.backward()
        if not torch.isfinite(weights.grad).all() or not torch.isfinite(bias.grad).all():
            raise ValueError("Native training gradient is nonfinite")
        optimizer.step()
    with torch.inference_mode():
        logits = (x @ weights + bias).numpy().copy()
    return {"weights": weights.detach().numpy().copy(), "bias": bias.detach().numpy().copy(),
            "losses": losses, "finalTrain": classification_metrics(logits, targets.tolist(), classes)}


def _model(payload, name):
    model = onnx.load_model_from_string(payload)  # Does not follow external_data.
    if model.functions or model.graph.sparse_initializer or model.training_info:
        raise ValueError(f"Unsupported {name} functions/sparse/training data")
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError(f"{name} contains external weights")
        if tensor.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
            if tensor.data_type != onnx.TensorProto.FLOAT or not np.isfinite(numpy_helper.to_array(tensor)).all():
                raise ValueError(f"{name} contains non-FP32 or nonfinite weights")
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g") or attribute.graphs or attribute.HasField("sparse_tensor") or attribute.sparse_tensors:
                raise ValueError(f"Unsupported {name} nested graph or sparse attribute")
            for tensor in ([attribute.t] if attribute.HasField("t") else []) + list(attribute.tensors):
                if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
                    raise ValueError(f"{name} contains external tensor attribute")
    if len({t.name for t in model.graph.initializer}) != len(model.graph.initializer):
        raise ValueError(f"Duplicate {name} initializer names")
    onnx.checker.check_model(model, full_check=True)
    return model


def _contract(value, name, shape):
    tensor = value.type.tensor_type
    actual = [dimension.dim_value if dimension.HasField("dim_value") else None for dimension in tensor.shape.dim]
    if value.name != name or tensor.elem_type != onnx.TensorProto.FLOAT or actual != shape:
        raise ValueError(f"Unexpected tensor contract: {name} {shape}")


def validate_export(source_bytes, model_bytes, head, labels):
    source, model = _model(source_bytes, "source"), _model(model_bytes, "export")
    if (source.ir_version != model.ir_version or list(source.opset_import) != list(model.opset_import)
            or list(source.graph.input) != list(model.graph.input)):
        raise ValueError("Export changed frozen model semantics or input")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Complete export must have exactly one image input and logits output")
    _contract(model.graph.input[0], "images", INPUT_SHAPE)
    classes, feature_size = head.get("classes"), head.get("featureSize")
    if (type(classes) is not int or type(feature_size) is not int or not 2 <= classes <= 8
            or not 1 <= feature_size <= 8192 or len(labels) != classes or len(set(labels)) != classes):
        raise ValueError("Invalid exported head dimensions or labels")
    _contract(model.graph.output[0], "logits", [1, classes])
    feature_info = next((v for v in source.graph.output if v.name == "features"), None)
    if feature_info is None:
        raise ValueError("Source features output missing")
    _contract(feature_info, "features", [1, feature_size])
    weight = finite_array(head.get("weights"), [feature_size * classes], "head weight", np.float32).reshape(feature_size, classes)
    bias = finite_array(head.get("bias"), [classes], "head bias", np.float32)
    originals = {value.name: value for value in source.graph.initializer}
    producers = {}
    for node in source.graph.node:
        for output in node.output:
            if output in producers:
                raise ValueError("Duplicate frozen tensor producer")
            producers[output] = node
    needed, nodes, visiting = set(), set(), set()
    inputs = {v.name for v in source.graph.input}

    def visit(name):
        if not name or name in needed:
            return
        if name in visiting:
            raise ValueError("Cyclic frozen graph")
        visiting.add(name)
        if name not in originals and name not in inputs:
            if name not in producers:
                raise ValueError("Missing frozen tensor")
            node = producers[name]
            for parent in node.input:
                visit(parent)
            nodes.add(id(node))
        needed.add(name)
        visiting.remove(name)

    visit("features")
    frozen_nodes = [node for node in source.graph.node if id(node) in nodes]
    # Keep exact parsed nodes: no operator, attribute, connection or constants may change.
    if len(model.graph.node) != len(frozen_nodes) + 2 or list(model.graph.node[:-2]) != frozen_nodes:
        raise ValueError("Export changed frozen graph nodes")
    linear, add = model.graph.node[-2:]
    if (linear.op_type != "MatMul" or list(linear.input) != ["features", "ovb.head.weight"]
            or list(linear.output) != ["ovb.head.matmul"] or linear.attribute or linear.domain
            or add.op_type != "Add" or list(add.input) != ["ovb.head.matmul", "ovb.head.bias"]
            or list(add.output) != ["logits"] or add.attribute or add.domain):
        raise ValueError("Exported head graph is not the expected linear classifier")
    exported = {value.name: value for value in model.graph.initializer}
    retained = {name for name in originals if name in needed}
    if set(exported) != retained | {"ovb.head.weight", "ovb.head.bias"}:
        raise ValueError("Export did not replace the source classifier and preserve all frozen weights")
    for name in retained:
        if exported[name] != originals[name]:
            raise ValueError(f"Export changed frozen initializer: {name}")
    for name, expected in [("ovb.head.weight", weight), ("ovb.head.bias", bias)]:
        actual = numpy_helper.to_array(exported[name])
        if actual.dtype != np.float32 or actual.shape != expected.shape or actual.astype("<f4").tobytes() != expected.astype("<f4").tobytes():
            raise ValueError(f"Exported head bytes differ: {name}")
    metadata = {value.key: value.value for value in model.metadata_props}
    if (len(metadata) != len(model.metadata_props) or json.loads(metadata.get("ovb.labels", "null")) != labels
            or metadata.get("ovb.recipe") != "mobilenetv4-small"):
        raise ValueError("Export labels/recipe metadata mismatch")
    return {"passed": True, "frozenInitializers": len(retained), "frozenNodes": len(frozen_nodes),
            "headParameters": feature_size * classes + classes, "modelSha256": sha256(model_bytes)}


def validate_plan(plan, dataset_sha):
    expected_training = {"featureSize": 1280, "classes": 5, "epochs": 40, "learningRate": .01,
                         "batchSize": 30, "initialization": "zeros", "optimizer": "sgd", "momentum": 0,
                         "weightDecay": 0, "shuffle": False, "augmentation": False, "backend": "wasm"}
    if (plan.get("schemaVersion") != 1 or plan.get("id") != "classification-flowers-60-v1"
            or plan.get("scope") != "classifier-head-only-public-flowers-v1" or plan.get("releaseReady") is not False
            or not re.fullmatch(r"[a-f0-9]{64}", dataset_sha) or plan.get("datasetManifestSha256") != dataset_sha
            or plan.get("classes") != ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
            or plan.get("splits") != {"train": 30, "tuning": 10, "evaluation": 20}
            or plan.get("training") != expected_training):
        raise ValueError("Changed/unbound classification plan, labels, split, or training recipe")
    model = plan.get("model", {})
    if (model.get("id") != MODEL_ID or model.get("sourceRevision") != SOURCE_REVISION
            or model.get("checkpointSha256") != SOURCE_SHA256
            or not isinstance(model.get("onnxSha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", model["onnxSha256"])):
        raise ValueError("Changed classification source model")
    if plan.get("preprocessing") != {"encodedInput": "canonical-rgba-png", "size": 224,
                                     "resize": "bilinear-half-pixel-stretch", "mean": [.485, .456, .406],
                                     "std": [.229, .224, .225], "layout": "NCHW", "dtype": "float32"}:
        raise ValueError("Changed preregistered preprocessing")
    if (plan.get("tolerances") != {"atol": ATOL, "rtol": RTOL, "inputMaxAbsoluteExclusive": 1e-6}
            or plan.get("criteria") != {"minimumTrainLossReduction": .2, "minimumTrainAccuracy": .9,
                                       "minimumEvaluationAccuracy": .7, "maximumEvaluationAccuracyGap": .05,
                                       "maximumEvaluationMacroF1Gap": .05}
            or plan.get("budget") != {"maxCompletedRuns": 1, "maxRuntimeSeconds": 600,
                                     "maxActiveWorkMinutes": 90, "automaticRetries": 0}):
        raise ValueError("Changed preregistered criteria, tolerances, or budget")


def validate_manifest(manifest, plan):
    images = manifest.get("images")
    if (manifest.get("schemaVersion") != 1 or manifest.get("classes") != plan["classes"]
            or not isinstance(images, list) or len(images) != 60):
        raise ValueError("Expected fixed 60-image manifest and class order")
    for item in images:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"])
                or type(item.get("label")) is not int or not 0 <= item["label"] < 5
                or item.get("split") not in ("train", "tuning", "evaluation")):
            raise ValueError("Invalid image ID, label or split")
    for key in ("id", "creatorKey", "flickrId", "originalSha256", "pngSha256"):
        values = [item.get(key) for item in images]
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != 60:
            raise ValueError(f"Duplicate or missing dataset identity: {key}")
    for label in range(5):
        if Counter(item["split"] for item in images if item["label"] == label) != {"train": 6, "tuning": 2, "evaluation": 4}:
            raise ValueError("Each class requires the fixed 6/2/4 split")
    return [item for item in images if item["split"] != "tuning"]


def validate_request(request, active, labels):
    if (request.get("labels") != labels or request.get("epochs") != 40
            or request.get("learningRate") != .01 or not isinstance(request.get("samples"), list)
            or len(request["samples"]) != len(active)):
        raise ValueError("Actual Worker request differs from the fixed recipe")
    identifiers = []
    for actual, expected in zip(request["samples"], active):
        if (actual.get("imageId") != expected["id"] or actual.get("name") != expected["id"] + ".png"
                or actual.get("label") != expected["label"]
                or actual.get("split") != ("train" if expected["split"] == "train" else "test")
                or actual.get("fileSha256") != expected["pngSha256"]
                or not isinstance(actual.get("id"), str) or not actual["id"]):
            raise ValueError("Actual Worker image order/label/split/encoded hash mismatch")
        identifiers.append(actual["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate actual Worker sample identity")


def validate_records(plan, active, browser, fresh, report, bindings, model_bytes):
    model_sha = sha256(model_bytes)
    for name, value in [("browser", browser), ("fresh browser", fresh)]:
        if value.get("schemaVersion") != 1 or any(value.get(key) != expected for key, expected in bindings.items()):
            raise ValueError(f"Unbound {name} evidence")
        if value.get("modelSha256") != model_sha:
            raise ValueError(f"{name} model digest mismatch")
    if (browser.get("sourceModelSha256") != plan["model"]["onnxSha256"]
            or browser.get("exportedModelBytes") != len(model_bytes) or browser.get("labels") != plan["classes"]):
        raise ValueError("Browser source model/labels/bytes mismatch")
    validate_request(browser.get("request", {}), active, plan["classes"])
    for name, rows in [("browser", browser.get("images")), ("fresh", fresh.get("images"))]:
        if not isinstance(rows, list) or [row.get("id") for row in rows] != [row["id"] for row in active]:
            raise ValueError(f"{name} image identity/order mismatch")
    head = browser.get("head", {})
    if head.get("classes") != 5 or head.get("featureSize") != 1280:
        raise ValueError("Unexpected final head shape")
    finite_array(head.get("weights"), [6400], "saved head weights", np.float32)
    finite_array(head.get("bias"), [5], "saved head bias", np.float32)
    summary, training = browser.get("summary", {}), report.get("training", {})
    losses = finite_array(summary.get("losses"), [40], "actual Worker losses")
    if (summary.get("epochs") != 40 or summary.get("trainingCount") != 30 or summary.get("testCount") != 20
            or summary.get("backend") != "wasm" or summary.get("modelSha256") != model_sha
            or summary.get("modelBytes") != len(model_bytes)):
        raise ValueError("Actual Worker summary metadata mismatch")
    if (report.get("schemaVersion") != 1 or report.get("labels") != plan["classes"]
            or report.get("artifact") != {"file": "model.onnx", "sha256": model_sha, "bytes": len(model_bytes)}
            or report.get("model", {}).get("sha256") != SOURCE_SHA256
            or report["model"].get("revision") != SOURCE_REVISION or report["model"].get("modelId") != MODEL_ID
            or training.get("scope") != "classifier-head-only" or training.get("epochsCompleted") != 40
            or training.get("learningRate") != .01 or training.get("backend") != "wasm"
            or training.get("engine") != "@jax-js/jax@0.1.27" or training.get("lossTiming") != "before-each-update"
            or not np.array_equal(finite_array(training.get("losses"), [40], "report losses"), losses)):
        raise ValueError("ZIP report training/model metadata or loss timing mismatch")
    verification = report.get("verification", {})
    if (verification.get("atol") != ATOL or verification.get("rtol") != RTOL
            or verification.get("native", {}).get("passed") is not True
            or verification.get("export", {}).get("passed") is not True
            or verification.get("export", {}).get("images") != 21):
        raise ValueError("Missing successful production Worker verification")
    for key in ("accuracy", "trainingCount", "testCount", "durationMs"):
        if report.get("metrics", {}).get(key) != summary.get(key):
            raise ValueError(f"Actual Worker/ZIP report metric mismatch: {key}")
    for key in ("durationMs", "nativeMaxAbsError", "exportMaxAbsError", "weightDeltaL2"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid Worker summary {key}")
    if summary["weightDeltaL2"] <= 0 or training.get("weightDeltaL2") != summary["weightDeltaL2"]:
        raise ValueError("Missing or inconsistent head update norm")
    requests = [item for item in browser["request"]["samples"] if item["split"] == "test"]
    predictions = summary.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 20:
        raise ValueError("Missing actual Worker held-out predictions")
    rows = {row["id"]: row for row in browser["images"]}
    checks, correct = [], 0
    for prediction, request in zip(predictions, requests):
        scores = finite_array(prediction.get("scores"), [5], "actual Worker scores")
        if (prediction.get("id") != request["id"] or prediction.get("actual") != request["label"]
                or prediction.get("predicted") != int(scores.argmax()) or np.any(scores < 0) or np.any(scores > 1)
                or abs(float(scores.sum()) - 1) > 1e-6):
            raise ValueError("Actual Worker prediction identity/label/score mismatch")
        correct += prediction["actual"] == prediction["predicted"]
        supplemental = classification_metrics([rows[request["imageId"]]["finalLogits"]], [request["label"]], 5)
        checks.append(compare_arrays(scores, supplemental["probabilities"][0], f"worker-score:{request['imageId']}"))
    if summary.get("accuracy") != correct / 20:
        raise ValueError("Actual Worker accuracy does not match its predictions")
    for row, item in zip(browser["images"], active):
        if row.get("label") != item["label"] or row.get("split") != item["split"]:
            raise ValueError("Supplemental image label/split mismatch")
        finite_array(row.get("features"), [1280], "browser features", np.float32)
        finite_array(row.get("baseLogits"), [1000], "browser base logits", np.float32)
        finite_array(row.get("finalLogits"), [5], "browser final logits", np.float32)
    for row in fresh["images"]:
        finite_array(row.get("logits"), [5], "fresh browser logits", np.float32)
    return checks


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_image_input(images, run, item, row):
    from PIL import Image
    image_id = item["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
        raise ValueError("Unsafe image identity")
    path = Path(images) / f"{image_id}.png"
    raw_path = Path(run) / "inputs" / f"{image_id}.f32"
    if path.is_symlink() or raw_path.is_symlink():
        raise ValueError("Symlink image/input is not allowed")
    png, payload = path.read_bytes(), raw_path.read_bytes()
    if len(png) != item["pngBytes"] or sha256(png) != item["pngSha256"]:
        raise ValueError(f"Canonical PNG bytes/hash mismatch: {image_id}")
    if len(payload) != 4 * math.prod(INPUT_SHAPE) or sha256(payload) != row.get("inputSha256"):
        raise ValueError(f"Supplemental browser input bytes/hash mismatch: {image_id}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGBA" or image.size != (item["width"], item["height"]):
            raise ValueError(f"Canonical PNG format/shape mismatch: {image_id}")
    browser_input = finite_array(np.frombuffer(payload, dtype="<f4").reshape(INPUT_SHAPE), INPUT_SHAPE, "browser input", np.float32)
    kit = _module(ROOT / "public/kit/predict.py", "ovb_public_classifier_kit")
    native_input = finite_array(kit.preprocess_image(path), INPUT_SHAPE, "native input", np.float32)
    error = float(np.max(np.abs(native_input.astype(np.float64) - browser_input.astype(np.float64))))
    if not error < 1e-6:
        raise ValueError(f"Independent preprocessing mismatch: {image_id}, maxAbsoluteError={error}")
    return native_input, {"id": image_id, "passed": True, "maxAbsoluteError": error,
                          "browserInputSha256": sha256(payload), "nativeInputSha256": sha256(native_input.astype("<f4").tobytes())}


def load_native(cache):
    checkpoint = Path(cache) / "model.safetensors"
    payload = checkpoint.read_bytes()
    if len(payload) != 15_223_016 or sha256(payload) != SOURCE_SHA256:
        raise ValueError("Pinned source checkpoint bytes/hash mismatch")
    versions = {"torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu", "timm": "1.0.15",
                "onnx": "1.17.0", "onnxruntime": "1.20.1", "numpy": "2.2.3",
                "safetensors": "0.5.3", "Pillow": "11.1.0"}
    for package, expected in versions.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Native reference requires pinned {package}=={expected}")
    import torch
    import timm
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    torch.manual_seed(0)
    state = load_file(str(checkpoint), device="cpu")
    if tuple(state["classifier.weight"].shape) != (1000, 1280) or tuple(state["conv_head.weight"].shape) != (1280, 960, 1, 1):
        raise ValueError("Native source classifier/conv_head shape mismatch")
    if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise ValueError("Source checkpoint has nonfinite values")
    base = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False)
    base.load_state_dict(state, strict=True)
    base.requires_grad_(False).eval().float()
    return base, versions


def quality_checks(browser, native):
    result = []
    for name, metrics in [("browser", browser), ("native", native)]:
        improvement = 1 - metrics["finalTrain"]["crossEntropy"] / metrics["initialTrain"]["crossEntropy"]
        for suffix, value, minimum in [("train-loss-reduction", improvement, .2),
                                       ("train-accuracy", metrics["finalTrain"]["accuracy"], .9),
                                       ("evaluation-accuracy", metrics["evaluation"]["accuracy"], .7)]:
            result.append({"name": f"{name}:{suffix}", "passed": bool(value >= minimum), "value": value, "minimum": minimum})
    for metric in ("accuracy", "macroF1"):
        gap = abs(browser["evaluation"][metric] - native["evaluation"][metric])
        # Preserve the inclusive five-point boundary despite float64 subtraction
        # (e.g. .8 - .75). This only absorbs arithmetic roundoff, not model error.
        within_limit = gap <= .05 or math.isclose(gap, .05, rel_tol=0, abs_tol=8 * sys.float_info.epsilon)
        result.append({"name": f"evaluation-{metric}-gap", "passed": bool(within_limit), "value": gap, "maximum": .05})
    return result


def _metric_sets(logits, active):
    train = [index for index, item in enumerate(active) if item["split"] == "train"]
    evaluation = [index for index, item in enumerate(active) if item["split"] == "evaluation"]
    labels = [item["label"] for item in active]
    return {"initialTrain": classification_metrics(np.zeros((len(train), 5)), [labels[i] for i in train], 5),
            "finalTrain": classification_metrics(logits[train], [labels[i] for i in train], 5),
            "initialEvaluation": classification_metrics(np.zeros((len(evaluation), 5)), [labels[i] for i in evaluation], 5),
            "evaluation": classification_metrics(logits[evaluation], [labels[i] for i in evaluation], 5)}


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_comparison(args, evidence):
    plan, plan_sha = read_json(args.plan)
    manifest, data_sha = read_json(args.data)
    validate_plan(plan, data_sha)
    active = validate_manifest(manifest, plan)
    bindings = {"planSha256": plan_sha, "datasetManifestSha256": data_sha}
    evidence.update(bindings)
    browser, _ = read_json(args.run / "browser.json")
    fresh, _ = read_json(args.run / "fresh-browser.json")
    report, _ = read_json(args.run / "report.json")
    model_bytes = (args.run / "model.onnx").read_bytes()
    source_bytes = args.source_model.read_bytes()
    if sha256(source_bytes) != plan["model"]["onnxSha256"]:
        raise ValueError("Prepared source ONNX differs from preregistration")
    checks = validate_records(plan, active, browser, fresh, report, bindings, model_bytes)
    evidence.update({"modelSha256": sha256(model_bytes), "sourceModelSha256": sha256(source_bytes),
                     "scope": plan["scope"], "classes": plan["classes"], "tolerances": plan["tolerances"],
                     "comparisons": checks, "inputs": []})
    evidence["exportIntegrity"] = validate_export(source_bytes, model_bytes, browser["head"], plan["classes"])
    atomic_json(args.run / "comparison.json", evidence)
    inputs = []
    for item, row in zip(active, browser["images"]):
        values, comparison = read_image_input(args.images, args.run, item, row)
        inputs.append(values)
        evidence["inputs"].append(comparison)
    # All independent inputs pass before native features/training are interpreted.
    atomic_json(args.run / "comparison.json", evidence)
    base, versions = load_native(args.source_cache)
    import torch
    import onnxruntime as ort
    before = _state_hash(base)
    native = {"schemaVersion": 1, **bindings, "status": "running", "sourceCheckpointSha256": SOURCE_SHA256,
              "environment": {"python": sys.version, "versions": versions, "threads": 1, "device": "cpu"},
              "codeSha256": {name: sha256((ROOT / name).read_bytes()) for name in
                             ["scripts/classification_real_reference.py", "public/kit/predict.py", "scripts/prepare_model.py"]},
              "images": []}
    for item, values, observed in zip(active, inputs, browser["images"]):
        with torch.inference_mode():
            features = base.forward_head(base.forward_features(torch.from_numpy(values)), pre_logits=True)
            base_logits = base.classifier(features)
        features_np = finite_array(features.numpy(), [1, 1280], "native features", np.float32)[0]
        base_np = finite_array(base_logits.numpy(), [1, 1000], "native base logits", np.float32)[0]
        native["images"].append({"id": item["id"], "label": item["label"], "split": item["split"],
                                 "features": features_np.tolist(), "baseLogits": base_np.tolist()})
        checks.append(compare_arrays(observed["features"], features_np, f"source-features:{item['id']}"))
        checks.append(compare_arrays(observed["baseLogits"], base_np, f"source-logits:{item['id']}"))
    atomic_json(args.run / "native.json", native)
    train_indices = [index for index, item in enumerate(active) if item["split"] == "train"]
    feature_matrix = np.asarray([row["features"] for row in native["images"]], dtype=np.float32)
    trained = train_linear(feature_matrix[train_indices], [active[index]["label"] for index in train_indices], 40, .01, 5)
    native["head"] = {"classes": 5, "featureSize": 1280, "weights": trained["weights"].ravel().tolist(), "bias": trained["bias"].tolist()}
    native["training"] = {"epochs": 40, "learningRate": .01, "lossTiming": "before-each-update", "losses": trained["losses"]}
    checks.append(compare_arrays(browser["summary"]["losses"], trained["losses"], "training-loss-trajectory"))
    checks.append(compare_arrays(browser["head"]["weights"], trained["weights"].ravel(), "final-head-weight"))
    checks.append(compare_arrays(browser["head"]["bias"], trained["bias"], "final-head-bias"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    native_logits, browser_logits, per_image = [], [], []
    browser_weight = np.asarray(browser["head"]["weights"], dtype=np.float64).reshape(1280, 5)
    browser_bias = np.asarray(browser["head"]["bias"], dtype=np.float64)
    for index, (item, values, observed, fresh_row) in enumerate(zip(active, inputs, browser["images"], fresh["images"])):
        with torch.inference_mode():
            output = (torch.from_numpy(feature_matrix[index:index + 1]) @ torch.from_numpy(trained["weights"]) + torch.from_numpy(trained["bias"])).numpy()[0]
        native_logits.append(output)
        native["images"][index]["finalLogits"] = output.tolist()
        actual = finite_array(observed["finalLogits"], [5], "browser logits", np.float32)
        browser_logits.append(actual)
        recomputed = (np.asarray(observed["features"], dtype=np.float64) @ browser_weight + browser_bias).astype(np.float32)
        cpu_logits = finite_array(cpu.run(["logits"], {"images": values})[0], [1, 5], "Python export logits", np.float32)[0]
        image_checks = [compare_arrays(actual, recomputed, f"saved-head-forward:{item['id']}"),
                        compare_arrays(actual, output, f"learned-native:{item['id']}"),
                        compare_arrays(fresh_row["logits"], actual, f"fresh-browser-export:{item['id']}"),
                        compare_arrays(cpu_logits, actual, f"python-export:{item['id']}")]
        checks.extend(image_checks)
        per_image.append({"id": item["id"], "split": item["split"], "actual": item["label"],
                          "actualLabel": plan["classes"][item["label"]], "browserPredicted": int(actual.argmax()),
                          "nativePredicted": int(output.argmax()), "pythonPredicted": int(cpu_logits.argmax()),
                          "freshPredicted": int(np.argmax(fresh_row["logits"])),
                          "browserLogits": actual.tolist(), "nativeLogits": output.tolist(),
                          "pythonLogits": cpu_logits.tolist(), "freshLogits": fresh_row["logits"],
                          "passed": all(value["passed"] for value in image_checks)})
    if _state_hash(base) != before:
        raise ValueError("Native frozen parameters changed during the reference")
    browser_metrics = _metric_sets(np.asarray(browser_logits), active)
    native_metrics = _metric_sets(np.asarray(native_logits), active)
    quality = quality_checks(browser_metrics, native_metrics)
    numeric_passed = all(item["passed"] for item in checks)
    native.update({"status": "completed", "metrics": native_metrics, "frozenStateSha256": before, "frozenUnchanged": True})
    atomic_json(args.run / "native.json", native)
    evidence.update({"status": "completed", "numericPassed": numeric_passed,
                     "qualityPassed": all(item["passed"] for item in quality),
                     "passed": numeric_passed and all(item["passed"] for item in quality),
                     "metrics": {"browser": browser_metrics, "native": native_metrics},
                     "qualityConditions": quality, "images": per_image,
                     "failures": [item for item in checks + quality if not item["passed"]],
                     "misclassifications": [item for item in per_image if item["browserPredicted"] != item["actual"] or item["nativePredicted"] != item["actual"]],
                     "nativeFrozenUnchanged": True, "nativeEvidenceSha256": sha256((args.run / "native.json").read_bytes()),
                     "evidenceBoundary": "Input/features/head-forward are supplemental production-function measurements; head bytes are extracted from ONNX. Actual Worker request, loss history, scores and report are separately bound."})
    atomic_json(args.run / "comparison.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "data", "images", "run", "source-model", "source-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evidence = {"schemaVersion": 1, "status": "running", "passed": False, "releaseReady": False}
    try:
        if not args.run.is_dir() or args.run.is_symlink():
            raise ValueError("Run must be an existing nonsymlink experiment directory")
        for name in ("native.json", "comparison.json"):
            if (args.run / name).exists():
                raise ValueError("Native/comparison output already exists; automatic reruns are forbidden")
        result = run_comparison(args, evidence)
    except Exception as error:
        evidence.update({"status": "error", "passed": False, "error": str(error)})
        # Preserve any preexisting report instead of replacing earlier evidence.
        target = args.run / "comparison.json"
        try:
            if target.exists() and not evidence.get("planSha256"):
                print(json.dumps(evidence, allow_nan=False), file=sys.stderr)
                return 1
            atomic_json(target, evidence)
        except Exception as save_error:
            evidence["reportSaveError"] = str(save_error)
        print(json.dumps(evidence, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "passed": result["passed"],
                      "numericPassed": result["numericPassed"], "qualityPassed": result["qualityPassed"],
                      "failures": len(result["failures"])}, allow_nan=False))
    return 0  # A completed scientific FAIL is evidence, not an infrastructure crash.


if __name__ == "__main__":
    raise SystemExit(main())
