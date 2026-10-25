#!/usr/bin/env python3
"""Bounded, local developer reference and controls for D-FINE own-prefix training.

No application server or inference service is created. The reference arm uses
its own PyTorch prefix; a control only reads browser features/weights and never
opens or advances the reference trajectory. Large tensors stay in artifacts/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"
TRAINABLE = [f"model.decoder.bbox_embed.2.layers.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")]
TRAINABLE += [f"model.decoder.class_embed.2.{kind}" for kind in ("weight", "bias")]
FROZEN = [f"model.decoder.lqe_layers.2.reg_conf.layers.{i}.{kind}" for i in range(2) for kind in ("weight", "bias")]
FEATURE_SHAPES = {"bboxInput": [300, 128], "classInput": [300, 128],
                  "residualCorners": [300, 132], "references": [300, 4]}
PREDICTION_SHAPES = {"logits": [300, 80], "boxes": [300, 4], "corners": [300, 132]}
INPUT_SHAPE = [1, 3, 640, 640]
LOSS_NAMES = ("vfl", "l1", "giou", "total")
LEARNING_RATE = 1e-4


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_hash(value):
    return sha256_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing a symlink report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.is_symlink():
        raise ValueError("Refusing a symlink temporary report")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def exact_keys(value, names, context):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{context} keys must be exactly {list(names)}")


def finite_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be finite numeric data")
    return value


def checked_tensor(value, shape, context):
    if (not isinstance(value, dict) or not isinstance(value.get("shape"), list)
            or any(type(n) is not int or n < 0 for n in value["shape"]) or value["shape"] != shape
            or not isinstance(value.get("data"), list) or len(value["data"]) != math.prod(shape)):
        raise ValueError(f"{context} must have exact shape {shape}")
    if any(isinstance(n, bool) or not isinstance(n, (float, int)) or not math.isfinite(n) for n in value["data"]):
        raise ValueError(f"{context} contains nonfinite or nonnumeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value["data"], dtype=np.float32).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError(f"{context} must remain finite in FP32")
    return array


def checked_targets(value):
    exact_keys(value, ("labels", "boxes"), "targets")
    labels, boxes = value["labels"], value["boxes"]
    if (not isinstance(labels, list) or any(type(n) is not int or not 0 <= n < 80 for n in labels)
            or not isinstance(boxes, list) or len(boxes) != 4 * len(labels)):
        raise ValueError("Targets must contain original 80-class indices and aligned boxes")
    for n in boxes:
        finite_number(n, "target box")
    array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if (not np.isfinite(array).all() or np.any(array[:, 2:] <= 0)
            or np.any(array[:, :2] - array[:, 2:] / 2 < -1e-6)
            or np.any(array[:, :2] + array[:, 2:] / 2 > 1 + 1e-6)):
        raise ValueError("Target boxes must be positive normalized cxcywh within the image")
    return np.asarray(labels, dtype=np.int64), array


def read_input(data_dir, item):
    image_id, spec = item.get("id"), item.get("input")
    if not isinstance(image_id, str) or not re.fullmatch(r"\d{12}", image_id) or not isinstance(spec, dict):
        raise ValueError("Invalid input identity")
    if (spec.get("file") != f"inputs/{image_id}.f32" or spec.get("shape") != INPUT_SHAPE
            or type(spec.get("bytes")) is not int or spec["bytes"] != 4 * math.prod(INPUT_SHAPE)
            or not isinstance(spec.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"])):
        raise ValueError("Invalid canonical input shape, bytes, path or hash")
    root = Path(data_dir).resolve()
    path = root / spec["file"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Canonical input escapes the data directory")
    payload = path.read_bytes()
    if len(payload) != spec["bytes"] or sha256_bytes(payload) != spec["sha256"]:
        raise ValueError(f"Canonical input bytes/hash mismatch: {image_id}")
    array = np.frombuffer(payload, dtype="<f4").reshape(INPUT_SHAPE).copy()
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("Canonical input must contain finite RGB/255")
    return array


def validate_plan(plan, manifest_hash):
    if (not isinstance(plan, dict) or plan.get("schemaVersion") != 1
            or plan.get("id") != "dfine-coco-own-prefix-impact-v1"
            or plan.get("datasetManifestSha256") != manifest_hash):
        raise ValueError("Preregistered plan must bind the exact dataset manifest hash")
    expected = {"classes": 80, "trainable": "final-pretrained-bbox-and-class-heads", "frozen": "prefix-and-LQE",
                "loss": "VFL + 5 * L1 + 2 * GIoU", "optimizer": "SGD", "learningRate": 1e-4,
                "momentum": 0, "weightDecay": 0, "epochs": 3, "steps": 12,
                "order": "manifest-train-order-repeated", "randomness": "none; eval prefix; no augmentation or shuffle"}
    if (plan.get("sourceCheckpointSha256") != SOURCE_SHA256 or plan.get("recipe") != expected
            or plan.get("splits") != {"train": 4, "tuning": 2, "evaluation": 4}
            or plan.get("controls") != {"checkpoints": [0, 6, 12], "image": "first-training-image", "tolerances": "detector-training-tolerances.json"}
            or plan.get("metrics") != {"scoreThreshold": .05, "maxDetections": 100, "iouThreshold": .5, "recallPoints": 101, "nms": False}
            or plan.get("detectorQualification") != "FAIL" or plan.get("releaseReady") is not False):
        raise ValueError("Unsupported experiment recipe or changed qualification scope")
    if plan.get("criteria") != {
        "minimumTrainLossReduction": .01, "maximumOwnLossRelativeGap": .01,
        "maximumEvaluationAp50Gap": .01, "maximumEvaluationAp50Drop": .02,
        "exportLogits": {"atol": .0002, "rtol": .0001}, "exportBoxes": {"atol": .00002, "rtol": .0001},
    }:
        raise ValueError("Changed preregistered impact criteria")


def load_experiment(plan_path, manifest_path, data_dir):
    plan_bytes, manifest_bytes = Path(plan_path).read_bytes(), Path(manifest_path).read_bytes()
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    manifest_hash = sha256_bytes(manifest_bytes)
    validate_plan(plan, manifest_hash)
    images = manifest.get("images") if isinstance(manifest, dict) else None
    if (manifest.get("schemaVersion") != 1 or not isinstance(images, list) or len(images) != 10
            or len({item.get("id") for item in images}) != 10
            or Counter(item.get("split") for item in images) != plan["splits"]):
        raise ValueError("Dataset must have ten unique images in fixed 4/2/4 splits")
    inputs = {}
    for item in images:
        checked_targets(item["targets"])
        inputs[item["id"]] = read_input(data_dir, item)
    bindings = {"planSha256": sha256_bytes(plan_bytes), "datasetManifestSha256": manifest_hash}
    return plan, manifest, inputs, bindings


def load_assets(assets, plan):
    assets = Path(assets)
    manifest = read_json(assets / "model.json")
    if manifest.get("source", {}).get("sha256") != plan["sourceCheckpointSha256"]:
        raise ValueError("Prepared checkpoint differs from preregistration")
    checked = {}
    for key, filename in (("headWeights", "head-weights.json"), ("detector.onnx", "detector.onnx"), ("frozen-prefix.onnx", "frozen-prefix.onnx")):
        spec = manifest["artifacts"][key]
        payload = (assets / filename).read_bytes()
        if spec.get("file") != filename or spec.get("bytes") != len(payload) or spec.get("sha256") != sha256_bytes(payload):
            raise ValueError(f"Prepared artifact hash/bytes mismatch: {filename}")
        checked[key] = {"sha256": sha256_bytes(payload), "bytes": len(payload)}
    head = read_json(assets / "head-weights.json")
    config = head.get("config", {})
    if (head.get("schemaVersion") != 1 or any(config.get(key) != expected for key, expected in
            {"queries": 300, "hiddenSize": 128, "classes": 80, "bins": 33, "regScale": 4, "topK": 4}.items())):
        raise ValueError("Prepared head config is not the fixed Nano 80-class recipe")
    checked_tensor({"shape": [33], "data": config.get("projection")}, [33], "projection")
    exact_keys(head.get("weights"), TRAINABLE + FROZEN, "prepared weights")
    for name, tensor in head["weights"].items():
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not shape or any(type(n) is not int or not 1 <= n <= 1024 for n in shape):
            raise ValueError(f"Invalid prepared parameter shape: {name}")
        checked_tensor(tensor, shape, name)
    return head, manifest, checked


def compare_array(actual, expected, bound):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Comparison requires matching finite nonempty tensor shapes")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowance = bound["absolute"] + bound["relative"] * np.abs(expected.astype(np.float64))
    failures = int(np.count_nonzero(error > allowance))
    return {"passed": failures == 0, "values": int(actual.size), "failures": failures,
            "maxAbsolute": float(error.max()), "maxRelative": float((error / np.maximum(np.abs(expected), 1e-12)).max()),
            "maxToleranceRatio": float((error / allowance).max()), "differenceL2": float(np.linalg.norm(error))}


def compare_measurements(actual, expected, tolerances):
    result = {"prediction": {}, "losses": {}, "gradients": {}, "nextWeights": {}, "frozenUnchanged": True}
    for group, names, limit in (("prediction", tuple(PREDICTION_SHAPES), "pretrainedForward"),
                               ("gradients", TRAINABLE, "pretrainedGradient"),
                               ("nextWeights", TRAINABLE + FROZEN, "pretrainedSgdWeights")):
        exact_keys(actual.get(group), names, group)
        exact_keys(expected.get(group), names, f"reference {group}")
        for name in names:
            shape = expected[group][name]["shape"]
            left = checked_tensor(actual[group][name], shape, f"{group}/{name}")
            right = checked_tensor(expected[group][name], shape, f"reference {group}/{name}")
            result[group][name] = compare_array(left, right, tolerances[limit])
            if group == "nextWeights" and name in FROZEN:
                result["frozenUnchanged"] &= left.tobytes() == right.tobytes()
    for side in (actual, expected):
        exact_keys(side.get("losses"), LOSS_NAMES, "losses")
        exact_keys(side.get("matches"), ("predictionIndices", "targetIndices"), "matching")
        lists = side["matches"]
        if (any(not isinstance(lists[key], list) or any(type(n) is not int or n < 0 for n in lists[key])
                or len(set(lists[key])) != len(lists[key]) for key in lists)
                or len(lists["predictionIndices"]) != len(lists["targetIndices"])):
            raise ValueError("Matching indices must be aligned distinct nonnegative integers")
    for name in LOSS_NAMES:
        left, right = finite_number(actual["losses"][name], name), finite_number(expected["losses"][name], name)
        result["losses"][name] = compare_array([left], [right], tolerances["pretrainedLoss"])
    result["matches"] = actual["matches"] == expected["matches"]
    result["passed"] = bool(result["matches"] and result["frozenUnchanged"] and all(
        row["passed"] for group in ("prediction", "losses", "gradients", "nextWeights") for row in result[group].values()))
    return result


def select_detections(prediction, score_threshold=.05, max_detections=100):
    logits = checked_tensor(prediction["logits"], [300, 80], "logits").ravel().astype(np.float64)
    boxes = checked_tensor(prediction["boxes"], [300, 4], "boxes")
    if np.any(boxes[:, 2:] < 0):
        raise ValueError("Predicted box sizes cannot be negative")
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp = np.exp(logits[~positive])
    scores[~positive] = exp / (1 + exp)
    selected = np.argsort(-scores, kind="stable")[:max_detections]
    return [{"queryIndex": int(index // 80), "classIndex": int(index % 80), "score": float(scores[index]),
             "box": boxes[index // 80].astype(np.float64).tolist()}
            for index in selected if scores[index] >= score_threshold]


def box_iou(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    low = np.maximum(left[:2] - left[2:] / 2, right[:2] - right[2:] / 2)
    high = np.minimum(left[:2] + left[2:] / 2, right[:2] + right[2:] / 2)
    intersection = float(np.maximum(high - low, 0).prod())
    union = float(left[2:].prod() + right[2:].prod() - intersection)
    return intersection / union if union > 0 else 0.0


def detection_metrics(images, detections):
    """101-point class AP50; global stable score order, greedy unmatched GT."""
    ids = [image["id"] for image in images]
    if len(set(ids)) != len(ids) or set(detections) != set(ids):
        raise ValueError("Detection metrics require exactly one prediction set per image")
    ground_truth = {}
    for image in images:
        labels, boxes = checked_targets(image["targets"])
        ground_truth[image["id"]] = (labels, boxes)
    by_class, per_class = {}, {}
    for ordinal, image_id in enumerate(ids):
        for entry in detections[image_id]:
            label = entry["classIndex"]
            if type(label) is not int or not 0 <= label < 80:
                raise ValueError("Invalid detection class index")
            finite_number(entry["score"], "detection score")
            if len(entry["box"]) != 4 or any(not math.isfinite(value) for value in entry["box"]):
                raise ValueError("Invalid detection box")
            by_class.setdefault(label, []).append((entry, image_id, ordinal))
    total_tp = total_fp = total_fn = 0
    for label in range(80):
        available = {image_id: [(index, box) for index, (gt_label, box) in enumerate(zip(*ground_truth[image_id])) if gt_label == label]
                     for image_id in ids}
        count = sum(len(items) for items in available.values())
        ranked = sorted(by_class.get(label, []), key=lambda row: (-row[0]["score"], row[2], row[0]["queryIndex"]))
        matched, true_positive = set(), []
        for entry, image_id, _ in ranked:
            candidates = [(box_iou(entry["box"], box), index) for index, box in available[image_id]
                          if (image_id, index) not in matched]
            overlap, index = max(candidates, key=lambda item: (item[0], -item[1])) if candidates else (0, -1)
            accepted = overlap >= .5
            true_positive.append(accepted)
            if accepted:
                matched.add((image_id, index))
        tp, fp = sum(true_positive), len(true_positive) - sum(true_positive)
        total_tp += tp
        total_fp += fp
        total_fn += count - tp
        if count:
            cumulative = np.cumsum(true_positive, dtype=np.float64)
            recall = cumulative / count
            precision = cumulative / np.arange(1, len(ranked) + 1)
            ap = float(np.mean([precision[recall >= level].max(initial=0) for level in np.linspace(0, 1, 101)]))
            per_class[str(label)] = {"ap50": ap, "groundTruth": count, "tp": tp, "fp": fp, "fn": count - tp}
    return {"ap50": float(np.mean([item["ap50"] for item in per_class.values()])) if per_class else None,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "classesWithGroundTruth": [int(label) for label in per_class], "perClass": per_class}


def load_module(filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_recipe(head):
    import importlib.metadata
    import torch
    for package, version in {"torch": "2.6.0+cpu", "numpy": "2.2.3", "scipy": "1.15.2"}.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Expected pinned {package}=={version} for the independent head reference")
    recipe = load_module("generate_detector_training_reference.py")
    if recipe.TRAINABLE != TRAINABLE or recipe.FROZEN != FROZEN:
        raise ValueError("Independent reference parameter scope changed")
    config = {**head["config"], "projection": torch.tensor(head["config"]["projection"], dtype=torch.float32)}
    weights = {name: torch.from_numpy(checked_tensor(tensor, tensor["shape"], name)).clone().requires_grad_(name in TRAINABLE)
               for name, tensor in head["weights"].items()}
    return torch, recipe, config, weights


def observe(features, weights, config, targets, torch, recipe):
    labels, boxes = checked_targets(targets)
    target = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(boxes)}
    with torch.no_grad():
        prediction = recipe.forward(features, weights, config)
        losses, matches = recipe.loss_and_match(prediction, target)
    result = {"prediction": {name: recipe.serial(value) for name, value in prediction.items()},
              "losses": {name: float(value) for name, value in losses.items()}, "matches": matches}
    for name, shape in PREDICTION_SHAPES.items():
        checked_tensor(result["prediction"][name], shape, name)
    for name in LOSS_NAMES:
        finite_number(result["losses"][name], name)
    return result


def measure_step(features, weights, config, targets, torch, recipe):
    labels, boxes = checked_targets(targets)
    target = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(boxes)}
    # Cloning keeps controls isolated from both the caller and the native arm.
    current = {name: value.detach().clone().requires_grad_(name in TRAINABLE) for name, value in weights.items()}
    prediction = recipe.forward(features, current, config)
    losses, matches = recipe.loss_and_match(prediction, target)
    losses["total"].backward()
    gradients = {}
    for name in TRAINABLE:
        value = current[name].grad
        if value is None or not torch.isfinite(value).all():
            raise ValueError(f"Missing or nonfinite actual gradient: {name}")
        gradients[name] = recipe.serial(value)
    result = {"weights": {name: recipe.serial(value) for name, value in current.items()},
              "prediction": {name: recipe.serial(value) for name, value in prediction.items()},
              "losses": {name: float(value.detach()) for name, value in losses.items()},
              "matches": matches, "gradients": gradients}
    with torch.no_grad():
        for name in TRAINABLE:
            current[name].sub_(current[name].grad, alpha=LEARNING_RATE)
            current[name].grad = None
    result["nextWeights"] = {name: recipe.serial(value) for name, value in current.items()}
    for name, shape in PREDICTION_SHAPES.items():
        checked_tensor(result["prediction"][name], shape, name)
    for name in LOSS_NAMES:
        finite_number(result["losses"][name], name)
    for name, tensor in result["nextWeights"].items():
        checked_tensor(tensor, list(weights[name].shape), name)
    return result, current


def reference(args):
    started = time.monotonic()
    plan, manifest, inputs, bindings = load_experiment(args.plan, args.manifest, args.data_dir)
    head, prepared, artifacts = load_assets(args.assets, plan)
    torch, recipe, config, weights = torch_recipe(head)
    prepare = load_module("prepare_detector.py")
    output = Path(args.output_dir)
    summary_path, trace_path = output / "summary.json", output / "reference.json"
    trace = {"schemaVersion": 1, "status": "running", **bindings, "source": prepared["source"],
             "preparedArtifacts": artifacts, "config": head["config"], "images": [],
             "initial": [], "final": [], "steps": [], "checkpoints": {"0": head["weights"]},
             "frozenUnchanged": False, "detectorQualification": "FAIL", "releaseReady": False}
    atomic_json(summary_path, {"schemaVersion": 1, "status": "running", **bindings})
    base, state, _aliases = prepare.load_source_model(args.cache)
    for name, saved in head["weights"].items():
        values = checked_tensor(saved, list(state[name].shape), name)
        if values.tobytes() != state[name].detach().numpy().astype("<f4").tobytes():
            raise ValueError(f"Prepared head does not equal the pinned checkpoint: {name}")
    inventory = lambda: {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "sha256": sha256_bytes(value.detach().numpy().tobytes())}
                         for name, value in base.state_dict().items()}
    original = inventory()
    wrapper = prepare.export_wrapper(base)
    features = {}
    for item in manifest["images"]:
        if time.monotonic() - started > 60 * plan["budget"]["runtimeMinutes"]:
            raise TimeoutError("Preregistered reference runtime budget exhausted")
        # no_grad produces ordinary tensors suitable for saving in head backward.
        with torch.no_grad():
            values = wrapper(torch.from_numpy(inputs[item["id"]]))
        own = {name: value[0].detach().clone() for name, value in zip(FEATURE_SHAPES, values[2:])}
        features[item["id"]] = own
        hashes = {}
        raw = {}
        for name, value in own.items():
            raw[name] = recipe.serial(value)
            array = checked_tensor(raw[name], FEATURE_SHAPES[name], name)
            hashes[name] = tensor_hash(array)
        atomic_json(output / "features" / f"{item['id']}.json", raw)
        trace["images"].append({"id": item["id"], "split": item["split"], "targets": item["targets"],
                                 "inputSha256": item["input"]["sha256"], "featureSha256": hashes})
        initial = observe(own, weights, config, item["targets"], torch, recipe)
        initial["sourceReconstruction"] = {name: compare_array(
            checked_tensor(initial["prediction"][name], PREDICTION_SHAPES[name], name), value[0].detach().numpy(),
            {"absolute": prepare.TOLERANCES[name]["atol"], "relative": prepare.TOLERANCES[name]["rtol"]})
            for name, value in zip(("logits", "boxes"), values[:2])}
        if not all(row["passed"] for row in initial["sourceReconstruction"].values()):
            atomic_json(output / "source-reconstruction-failure.json", {"id": item["id"], **initial})
            raise ValueError("Native reconstructed head differs from its complete source model")
        trace["initial"].append({"id": item["id"], **initial})
        atomic_json(trace_path, trace)
    train = [item for item in manifest["images"] if item["split"] == "train"]
    for index in range(plan["recipe"]["steps"]):
        if time.monotonic() - started > 60 * plan["budget"]["runtimeMinutes"]:
            raise TimeoutError("Preregistered reference runtime budget exhausted")
        item = train[index % len(train)]
        measurement, weights = measure_step(features[item["id"]], weights, config, item["targets"], torch, recipe)
        trace["steps"].append({"step": index + 1, "id": item["id"], **measurement})
        if index + 1 in plan["controls"]["checkpoints"]:
            trace["checkpoints"][str(index + 1)] = measurement["nextWeights"]
        atomic_json(trace_path, trace)
    for item in manifest["images"]:
        trace["final"].append({"id": item["id"], **observe(features[item["id"]], weights, config, item["targets"], torch, recipe)})
    after = inventory()
    trace["frozenUnchanged"] = original == after and all(
        tensor_hash(weights[name].detach().numpy()) == tensor_hash(checked_tensor(head["weights"][name], list(weights[name].shape), name)) for name in FROZEN)
    atomic_json(output / "source-state-hashes.json", {"before": original, "after": after})
    if not trace["frozenUnchanged"]:
        raise ValueError("Native frozen source or LQE changed")
    trace.update(status="completed", durationSeconds=time.monotonic() - started,
                 runtime={"torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads()})
    atomic_json(trace_path, trace)
    summary = {"schemaVersion": 1, "status": "completed", **bindings, "steps": len(trace["steps"]),
               "frozenUnchanged": trace["frozenUnchanged"], "durationSeconds": trace["durationSeconds"],
               "trace": {"file": "reference.json", "sha256": sha256_bytes(trace_path.read_bytes()), "bytes": trace_path.stat().st_size},
               "detectorQualification": "FAIL", "releaseReady": False}
    atomic_json(summary_path, summary)
    return summary


def control(args):
    plan, manifest, _inputs, bindings = load_experiment(args.plan, args.manifest, args.data_dir)
    head, _prepared, artifacts = load_assets(args.assets, plan)
    browser = read_json(args.browser)
    first = next(item for item in manifest["images"] if item["split"] == "train")
    if (browser.get("schemaVersion") != 1 or browser.get("id") != first["id"]
            or type(browser.get("step")) is not int or browser["step"] not in plan["controls"]["checkpoints"]
            or any(browser.get(key) != value for key, value in bindings.items())
            or browser.get("inputSha256") != first["input"]["sha256"] or browser.get("config") != head["config"]):
        raise ValueError("Control identity, checkpoint, plan, manifest, input or config mismatch")
    exact_keys(browser.get("features"), FEATURE_SHAPES, "control features")
    exact_keys(browser.get("weights"), TRAINABLE + FROZEN, "control weights")
    torch, recipe, config, _weights = torch_recipe(head)
    features = {name: torch.from_numpy(checked_tensor(browser["features"][name], shape, name)).clone()
                for name, shape in FEATURE_SHAPES.items()}
    weights = {}
    for name in TRAINABLE + FROZEN:
        array = checked_tensor(browser["weights"][name], head["weights"][name]["shape"], name)
        if ((name in FROZEN or browser["step"] == 0)
                and array.tobytes() != checked_tensor(head["weights"][name], list(array.shape), name).tobytes()):
            raise ValueError(f"Control changed initial or fixed LQE tensor: {name}")
        weights[name] = torch.from_numpy(array).clone().requires_grad_(name in TRAINABLE)
    expected, _unused_candidate = measure_step(features, weights, config, first["targets"], torch, recipe)
    raw_path = Path(args.output).with_name(Path(args.output).stem + "-reference.json")
    atomic_json(raw_path, {"schemaVersion": 1, **bindings, "id": first["id"], "step": browser["step"], **expected})
    tolerances = read_json(ROOT / "tests/fixtures/detector-training-tolerances.json")
    result = {"schemaVersion": 1, **bindings, "id": first["id"], "step": browser["step"],
              "scope": "Same actual browser features and browser weights; no native trajectory mutation",
              "preparedArtifacts": artifacts, "browserMeasurementSha256": sha256_bytes(Path(args.browser).read_bytes()),
              "featureSha256": {name: tensor_hash(value.detach().numpy()) for name, value in features.items()},
              "weightSha256": {name: tensor_hash(value.detach().numpy()) for name, value in weights.items()},
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads()},
              "reference": {"file": raw_path.name, "sha256": sha256_bytes(raw_path.read_bytes()), "bytes": raw_path.stat().st_size},
              "tolerances": tolerances, **compare_measurements(browser, expected, tolerances)}
    atomic_json(args.output, result)
    return result


def summarize_comparison(browser_trace, reference_trace, plan):
    """Pure own-path summary. Controls and export parity are separate gates."""
    validate_plan(plan, reference_trace.get("datasetManifestSha256"))
    for key in ("planSha256", "datasetManifestSha256"):
        if (not isinstance(reference_trace.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", reference_trace[key])
                or browser_trace.get(key) != reference_trace[key]):
            raise ValueError(f"Trace {key} binding mismatch")
    images = reference_trace.get("images", [])
    ids = [item["id"] for item in images]
    if len(ids) != 10 or len(set(ids)) != 10 or Counter(item["split"] for item in images) != plan["splits"]:
        raise ValueError("Reference trace has incomplete image splits")
    identity = lambda item: {key: item[key] for key in ("id", "split", "targets", "inputSha256")}
    if [identity(item) for item in browser_trace.get("images", [])] != [identity(item) for item in images]:
        raise ValueError("Browser/reference data identities differ")
    train_ids = [item["id"] for item in images if item["split"] == "train"]
    for trace in (browser_trace, reference_trace):
        for stage in ("initial", "final"):
            if [item.get("id") for item in trace.get(stage, [])] != ids:
                raise ValueError(f"Incomplete or reordered {stage} observations")
        if (len(trace.get("steps", [])) != 12 or any(item.get("step") != i + 1 or item.get("id") != train_ids[i % 4]
                                                    for i, item in enumerate(trace["steps"]))):
            raise ValueError("Incomplete, duplicated or out-of-order training steps")
        if set(trace.get("checkpoints", {})) != {"0", "6", "12"}:
            raise ValueError("Missing predetermined weight checkpoint")
        initial = trace["checkpoints"]["0"]
        exact_keys(initial, TRAINABLE + FROZEN, "initial checkpoint")
        shapes = {name: tensor["shape"] for name, tensor in initial.items()}
        initial_arrays = {name: checked_tensor(tensor, shapes[name], name) for name, tensor in initial.items()}
        for step in trace["steps"]:
            exact_keys(step.get("nextWeights"), TRAINABLE + FROZEN, "updated parameter")
            for name in TRAINABLE + FROZEN:
                current = checked_tensor(step["nextWeights"][name], shapes[name], name)
                if name in FROZEN and current.tobytes() != initial_arrays[name].tobytes():
                    raise ValueError(f"Actual trajectory changed frozen LQE: {name}")
            if step["step"] in (6, 12):
                checkpoint = trace["checkpoints"][str(step["step"])]
                exact_keys(checkpoint, TRAINABLE + FROZEN, "checkpoint")
                if any(checked_tensor(checkpoint[name], shapes[name], name).tobytes()
                       != checked_tensor(step["nextWeights"][name], shapes[name], name).tobytes()
                       for name in TRAINABLE + FROZEN):
                    raise ValueError("Saved checkpoint differs from the accepted update")
    for name in TRAINABLE + FROZEN:
        expected = reference_trace["checkpoints"]["0"][name]
        if checked_tensor(browser_trace["checkpoints"]["0"][name], expected["shape"], name).tobytes() != checked_tensor(expected, expected["shape"], name).tobytes():
            raise ValueError("Both trajectories must start at identical pretrained weights")
    tolerances = read_json(ROOT / "tests/fixtures/detector-training-tolerances.json")
    steps, own_loss_passed = [], True
    for actual, expected in zip(browser_trace["steps"], reference_trace["steps"]):
        comparison = compare_measurements(actual, expected, tolerances)
        gaps = {name: abs(actual["losses"][name] - expected["losses"][name]) / max(abs(expected["losses"][name]), 1e-8) for name in LOSS_NAMES}
        own_loss_passed &= all(value <= plan["criteria"]["maximumOwnLossRelativeGap"] for value in gaps.values())
        steps.append({"step": actual["step"], "id": actual["id"], "lossRelativeGap": gaps,
                      "comparison": comparison})
    arms, outputs = {}, {}
    for name, trace in (("browser", browser_trace), ("native", reference_trace)):
        stages = {}
        for stage in ("initial", "final"):
            by_id = {item["id"]: item for item in trace[stage]}
            splits = {}
            for split in plan["splits"]:
                selected = [item for item in images if item["split"] == split]
                detections = {item["id"]: select_detections(by_id[item["id"]]["prediction"]) for item in selected}
                losses = {loss: float(np.mean([finite_number(by_id[item["id"]]["losses"][loss], loss) for item in selected])) for loss in LOSS_NAMES}
                splits[split] = {"meanLoss": losses, "detection": detection_metrics(selected, detections), "detections": detections}
            stages[stage] = splits
        before = stages["initial"]["train"]["meanLoss"]["total"]
        after = stages["final"]["train"]["meanLoss"]["total"]
        reduction = (before - after) / max(abs(before), 1e-8)
        ap_before = stages["initial"]["evaluation"]["detection"]["ap50"]
        ap_after = stages["final"]["evaluation"]["detection"]["ap50"]
        stages["trainLossReduction"] = reduction
        stages["trainImproved"] = reduction >= plan["criteria"]["minimumTrainLossReduction"]
        stages["evaluationAp50Drop"] = None if ap_before is None or ap_after is None else ap_before - ap_after
        stages["frozenUnchanged"] = trace.get("frozenUnchanged") is True
        change = {"localization": 0.0, "classification": 0.0}
        for parameter in TRAINABLE:
            initial = trace["checkpoints"]["0"][parameter]
            final = trace["checkpoints"]["12"][parameter]
            left = checked_tensor(initial, initial["shape"], parameter)
            right = checked_tensor(final, initial["shape"], parameter)
            group = "localization" if ".bbox_embed." in parameter else "classification"
            change[group] += float(np.square(right.astype(np.float64) - left.astype(np.float64)).sum())
        stages["parameterDeltaL2"] = {key: math.sqrt(value) for key, value in change.items()}
        stages["bothHeadsUpdated"] = all(value > 0 for value in change.values())
        arms[name] = stages
    for stage in ("initial", "final"):
        rows = []
        for actual, expected in zip(browser_trace[stage], reference_trace[stage]):
            comparison = {}
            for name, bound_name in (("logits", "exportLogits"), ("boxes", "exportBoxes")):
                limit = plan["criteria"][bound_name]
                comparison[name] = compare_array(checked_tensor(actual["prediction"][name], PREDICTION_SHAPES[name], name),
                                                 checked_tensor(expected["prediction"][name], PREDICTION_SHAPES[name], name),
                                                 {"absolute": limit["atol"], "relative": limit["rtol"]})
            rows.append({"id": actual["id"], "comparison": comparison, "matches": actual["matches"] == expected["matches"]})
        outputs[stage] = rows
    ap_browser = arms["browser"]["final"]["evaluation"]["detection"]["ap50"]
    ap_native = arms["native"]["final"]["evaluation"]["detection"]["ap50"]
    ap_gap = None if ap_browser is None or ap_native is None else abs(ap_browser - ap_native)
    gates = {"bothArmsImprove": all(arm["trainImproved"] for arm in arms.values()),
             "ownLossTrajectory": own_loss_passed,
             "evaluationAp50Gap": ap_gap is not None and ap_gap <= plan["criteria"]["maximumEvaluationAp50Gap"],
             "evaluationAp50Drops": all(arm["evaluationAp50Drop"] is not None and arm["evaluationAp50Drop"] <= plan["criteria"]["maximumEvaluationAp50Drop"] for arm in arms.values()),
             "frozenUnchanged": all(arm["frozenUnchanged"] for arm in arms.values()),
             "bothHeadsUpdated": all(arm["bothHeadsUpdated"] for arm in arms.values())}
    passed = all(gates.values())
    decision = "continue-candidate" if passed else ("inconclusive-reference-did-not-improve" if not arms["native"]["trainImproved"] else "own-path-impact-outside-preregistered-conditions")
    return {"schemaVersion": 1, "scope": "Own-path impact only; controls and complete ONNX verification remain separate gates",
            "planSha256": reference_trace["planSha256"], "datasetManifestSha256": reference_trace["datasetManifestSha256"],
            "passed": passed, "decision": decision, "gates": gates, "arms": arms, "steps": steps,
            "rawOutputsOriginalTolerances": outputs, "evaluationAp50Gap": ap_gap,
            "matchingAgreementSteps": sum(row["comparison"]["matches"] for row in steps),
            "detectorQualification": "FAIL", "releaseReady": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "control", "finalize"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/p1d/data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/fixtures/detector-impact-data.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "tests/fixtures/detector-impact-plan.json")
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/p1b/detector-assets/unqualified")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/p1b/detector-assets/source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/p1d/reference")
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode != "reference" and (args.browser is None or args.output is None):
        parser.error("control/finalize require --browser and --output")
    destination = args.output_dir / "summary.json" if args.mode == "reference" else args.output
    try:
        atomic_json(destination, {"schemaVersion": 1, "status": "running", "mode": args.mode})
        if args.mode == "reference":
            result = reference(args)
        elif args.mode == "control":
            result = control(args)
        else:
            plan, _manifest, _inputs, bindings = load_experiment(args.plan, args.manifest, args.data_dir)
            native, browser = read_json(args.output_dir / "reference.json"), read_json(args.browser)
            if any(native.get(key) != value or browser.get(key) != value for key, value in bindings.items()):
                raise ValueError("Final trace bindings differ from current preregistration")
            result = summarize_comparison(browser, native, plan)
            atomic_json(destination, result)
    except Exception as error:
        result = {"schemaVersion": 1, "status": "error", "mode": args.mode, "passed": False,
                  "error": str(error), "detectorQualification": "FAIL", "releaseReady": False}
        atomic_json(destination, result)
    print(json.dumps({key: result[key] for key in ("status", "passed", "decision", "error") if key in result}, allow_nan=False))
    return 1 if result.get("passed") is False or result.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
