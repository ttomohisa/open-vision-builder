#!/usr/bin/env python3
"""Verify the bounded impact experiment's complete browser ONNX export offline.

Only canonical public inputs from the checksummed manifest are evaluated. This
component parity result never changes the existing detector qualification FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from verify_detector import (INPUT_SHAPE, OUTPUT_SHAPES, TOLERANCES, expected_tensor,
                             require_embedded_tensors, require_tensor_contract,
                             tensor_sha256)

INPUT_BYTES = 4 * int(np.prod(INPUT_SHAPE))
STATUS = {"schemaVersion": 1, "releaseReady": False, "detectorQualification": "FAIL"}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked_digest(value, expected: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or value != expected:
        raise ValueError(f"{name} SHA-256 mismatch")


def read_json(path: Path) -> tuple[dict, str]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    payload = Path(path).read_bytes()
    result = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return result, sha256(payload)


def require_schema(value: dict, name: str) -> None:
    if type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1:
        raise ValueError(f"Expected {name} schemaVersion 1")


def checked_tensor(value, shape: list[int], name: str) -> np.ndarray:
    if (not isinstance(value, dict) or value.get("shape") != shape
            or any(type(dimension) is not int for dimension in value["shape"])):
        raise ValueError(f"Expected {name} tensor with shape {shape}")
    return expected_tensor(value, "data", shape)


def read_input(data_dir: Path, item: dict) -> np.ndarray:
    spec = item.get("input")
    image_id = item.get("id")
    if not isinstance(image_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
        raise ValueError("Invalid manifest image id")
    if (not isinstance(spec, dict) or spec.get("shape") != INPUT_SHAPE
            or any(type(dimension) is not int for dimension in spec["shape"])
            or type(spec.get("bytes")) is not int or spec["bytes"] != INPUT_BYTES
            or spec.get("file") != f"inputs/{image_id}.f32"):
        raise ValueError(f"Invalid canonical input shape, bytes, or path: {image_id}")
    root = Path(data_dir).resolve()
    path = (root / spec["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Canonical input path escapes data directory: {image_id}")
    if path.stat().st_size != INPUT_BYTES:
        raise ValueError(f"Canonical input byte length mismatch: {image_id}")
    payload = path.read_bytes()
    checked_digest(spec.get("sha256"), sha256(payload), f"Input {image_id}")
    values = np.frombuffer(payload, dtype="<f4").reshape(INPUT_SHAPE)
    if not np.isfinite(values).all():
        raise ValueError(f"Canonical input must be finite: {image_id}")
    return values


def read_model(payload: bytes, name: str) -> onnx.ModelProto:
    # Never follow external_data locations. This shared recursive guard also
    # inspects tensor attributes and function/subgraph tensors before checking.
    model = onnx.load_model_from_string(payload)
    require_embedded_tensors(model)
    graph = model.graph
    if model.functions or graph.sparse_initializer:
        raise ValueError(f"Unsupported {name} functions or sparse initializers")
    if len(graph.input) != 1:
        raise ValueError(f"{name} must have exactly one image input")
    require_tensor_contract(graph.input[0], "images", INPUT_SHAPE)
    if len(graph.output) != 2 or {value.name for value in graph.output} != set(OUTPUT_SHAPES):
        raise ValueError(f"{name} must have exactly logits and boxes outputs")
    for output in graph.output:
        require_tensor_contract(output, output.name, OUTPUT_SHAPES[output.name])
    if len({value.name for value in graph.initializer}) != len(graph.initializer):
        raise ValueError(f"Duplicate {name} initializer names")
    try:
        onnx.checker.check_model(model, full_check=True)
    except onnx.checker.ValidationError as error:
        raise ValueError(f"Invalid {name} ONNX: {error}") from error
    return model


def verify_weights(source: onnx.ModelProto, model: onnx.ModelProto, recipe: dict,
                   checkpoint: dict) -> dict:
    require_schema(recipe, "export recipe")
    if (recipe.get("id") != "dfine-nano-coco-final-head-v1"
            or recipe.get("input") != {"name": "images", "shape": INPUT_SHAPE}
            or recipe.get("outputs") != [{"name": name, "shape": shape}
                                          for name, shape in OUTPUT_SHAPES.items()]):
        raise ValueError("Unexpected complete detector export recipe")
    if [node.SerializeToString() for node in source.graph.node] != [node.SerializeToString() for node in model.graph.node]:
        raise ValueError("Export changed graph nodes")
    if (source.ir_version != model.ir_version
            or list(source.opset_import) != list(model.opset_import)):
        raise ValueError("Export changed ONNX operator semantics")
    originals = {value.name: value for value in source.graph.initializer}
    exported = {value.name: value for value in model.graph.initializer}
    if originals.keys() != exported.keys():
        raise ValueError("Export changed initializer keys")
    bindings = recipe.get("parameters")
    if not isinstance(bindings, list) or not bindings or not isinstance(checkpoint, dict):
        raise ValueError("Missing trainable mappings or final browser checkpoint")
    mapped, parameters, hashes = set(), set(), {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("Invalid export parameter binding")
        parameter, names, shape = binding.get("parameter"), binding.get("initializers"), binding.get("shape")
        if (not isinstance(parameter, str) or not parameter or parameter in parameters
                or not isinstance(names, list) or not names
                or not isinstance(shape, list) or not shape
                or any(type(value) is not int or value < 1 for value in shape)):
            raise ValueError("Invalid or duplicate export parameter mapping")
        parameters.add(parameter)
        update = checked_tensor(checkpoint.get(parameter), shape, parameter)
        expected_bytes = np.ascontiguousarray(update, dtype="<f4").tobytes()
        for name in names:
            if not isinstance(name, str) or not name or name in mapped or name not in originals:
                raise ValueError(f"Missing or conflicting initializer mapping: {parameter}")
            mapped.add(name)
            for tensor in (originals[name], exported[name]):
                if tensor.data_type != onnx.TensorProto.FLOAT or list(tensor.dims) != shape:
                    raise ValueError(f"Mapped initializer shape/type mismatch: {name}")
            actual = numpy_helper.to_array(exported[name])
            if not np.isfinite(actual).all() or np.ascontiguousarray(actual, dtype="<f4").tobytes() != expected_bytes:
                raise ValueError(f"Exported initializer does not match final browser weights: {name}")
        hashes[parameter] = sha256(expected_bytes)
    frozen = originals.keys() - mapped
    for name in frozen:
        if originals[name].SerializeToString() != exported[name].SerializeToString():
            raise ValueError(f"Export changed frozen initializer: {name}")
    return {"frozenInitializerCount": len(frozen), "mappedInitializerCount": len(mapped),
            "trainableParameterCount": len(parameters), "finalParameterSha256": hashes,
            "graphNodesUnchanged": True, "initializerKeysUnchanged": True}


def indexed_predictions(rows, image_ids: list[str], name: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"Missing {name} predictions")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in image_ids or row["id"] in result:
            raise ValueError(f"Unexpected or duplicate {name} image id")
        prediction = row.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"Missing {name} prediction")
        result[row["id"]] = {key: checked_tensor(prediction.get(key), shape[1:], f"{name} {row['id']} {key}")
                             for key, shape in OUTPUT_SHAPES.items()}
    if set(result) != set(image_ids):
        raise ValueError(f"Missing {name} image predictions")
    return result


def compare(actual: dict, expected: dict) -> dict:
    result = {}
    for name in OUTPUT_SHAPES:
        reference = expected[name].astype(np.float64)
        errors = np.abs(actual[name].astype(np.float64) - reference)
        limits = TOLERANCES[name]
        allowance = limits["atol"] + limits["rtol"] * np.abs(reference)
        result[name] = {"passed": bool(np.all(errors <= allowance)), "shape": list(reference.shape),
                        "maxAbsoluteError": float(errors.max(initial=0)),
                        "maxToleranceRatio": float((errors / allowance).max(initial=0)),
                        "mismatchedValues": int(np.count_nonzero(errors > allowance)),
                        "values": int(reference.size)}
    return result


def verify(model_path: Path, source_path: Path, manifest_path: Path, plan_path: Path,
           data_dir: Path, browser_path: Path, fresh_browser_path: Path, recipe_path: Path) -> dict:
    manifest, manifest_hash = read_json(manifest_path)
    plan, plan_hash = read_json(plan_path)
    browser, browser_hash = read_json(browser_path)
    fresh, fresh_hash = read_json(fresh_browser_path)
    recipe, recipe_hash = read_json(recipe_path)
    for name, document in [("manifest", manifest), ("plan", plan), ("browser", browser),
                           ("fresh browser", fresh), ("model recipe", recipe)]:
        require_schema(document, name)
    checked_digest(plan.get("datasetManifestSha256"), manifest_hash, "Preregistered manifest")
    checked_digest(browser.get("datasetManifestSha256"), manifest_hash, "Browser manifest")
    checked_digest(browser.get("planSha256"), plan_hash, "Browser plan")
    criteria = plan.get("criteria", {})
    if (plan.get("id") != "dfine-coco-own-prefix-impact-v1"
            or criteria.get("exportLogits") != TOLERANCES["logits"]
            or criteria.get("exportBoxes") != TOLERANCES["boxes"]):
        raise ValueError("Plan must retain the original detector export tolerances")
    checked_digest(plan.get("sourceCheckpointSha256"), recipe.get("source", {}).get("sha256"), "Source checkpoint")
    model_bytes, source_bytes = Path(model_path).read_bytes(), Path(source_path).read_bytes()
    model_hash, source_hash = sha256(model_bytes), sha256(source_bytes)
    source_artifact = recipe.get("artifacts", {}).get("detector.onnx", {})
    checked_digest(source_artifact.get("sha256"), source_hash, "Source model")
    if type(source_artifact.get("bytes")) is not int or source_artifact["bytes"] != len(source_bytes):
        raise ValueError("Source model byte length mismatch")
    artifact = browser.get("artifact", {})
    checked_digest(artifact.get("sha256"), model_hash, "Browser export")
    if type(artifact.get("bytes")) is not int or artifact["bytes"] != len(model_bytes):
        raise ValueError("Browser export byte length mismatch")
    checked_digest(fresh.get("modelSha256"), model_hash, "Fresh browser export")
    source = read_model(source_bytes, "source model")
    model = read_model(model_bytes, "exported model")
    weights = verify_weights(source, model, recipe.get("exportRecipe", {}), browser.get("checkpoints", {}).get("12"))
    images = manifest.get("images")
    if not isinstance(images, list) or not images or any(not isinstance(item, dict) for item in images):
        raise ValueError("Manifest must contain canonical images")
    image_ids = [item.get("id") for item in images]
    if any(not isinstance(value, str) for value in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Manifest image ids must be distinct strings")
    final_predictions = indexed_predictions(browser.get("final"), image_ids, "final browser")
    fresh_predictions = indexed_predictions(fresh.get("images"), image_ids, "fresh browser")
    inputs = {item["id"]: read_input(data_dir, item) for item in images}
    input_hashes = {item["id"]: item["input"]["sha256"] for item in images}
    for row in fresh["images"]:
        checked_digest(row.get("inputSha256"), input_hashes[row["id"]], f"Fresh browser input {row['id']}")
    if ort.__version__ != "1.20.1":
        raise ValueError("Impact export verification requires ONNX Runtime 1.20.1")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    comparisons = []
    for item in images:
        image_id = item["id"]
        outputs = session.run(["logits", "boxes"], {"images": inputs[image_id]})
        actual = {}
        for name, output in zip(OUTPUT_SHAPES, outputs):
            if list(output.shape) != OUTPUT_SHAPES[name] or output.dtype != np.float32 or not np.isfinite(output).all():
                raise ValueError(f"Invalid CPU output {name}: {image_id}")
            actual[name] = output[0]
        comparison = {"cpuVsBrowserFinal": compare(actual, final_predictions[image_id]),
                      "cpuVsFreshBrowser": compare(actual, fresh_predictions[image_id]),
                      "freshBrowserVsBrowserFinal": compare(fresh_predictions[image_id], final_predictions[image_id])}
        comparisons.append({"id": image_id, "inputSha256": input_hashes[image_id],
                            "passed": all(metric["passed"] for pair in comparison.values() for metric in pair.values()),
                            "outputSha256": {name: tensor_sha256(value) for name, value in actual.items()},
                            "comparison": comparison})
    return {**STATUS, "passed": all(item["passed"] for item in comparisons),
            "scope": "Complete export invariance and raw-output parity on canonical public inputs",
            "modelSha256": model_hash, "modelBytes": len(model_bytes), "sourceModelSha256": source_hash,
            "datasetManifestSha256": manifest_hash, "planSha256": plan_hash, "recipeSha256": recipe_hash,
            "browserTraceSha256": browser_hash, "freshBrowserTraceSha256": fresh_hash,
            "inputName": "images", "inputShape": INPUT_SHAPE, "tolerances": TOLERANCES,
            "executionProvider": "CPUExecutionProvider", "onnxruntimeVersion": ort.__version__,
            "onnxVersion": onnx.__version__, "numpyVersion": np.__version__,
            **weights, "images": comparisons}


def atomic_report(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "source", "manifest", "plan", "data-dir", "browser", "fresh-browser", "recipe", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    # Invalidate any previous success before parsing inputs or starting ORT.
    atomic_report(args.output, json.dumps({**STATUS, "passed": False, "error": "Verification incomplete"}) + "\n")
    try:
        result = verify(args.model, args.source, args.manifest, args.plan, args.data_dir,
                        args.browser, args.fresh_browser, args.recipe)
    except Exception as error:
        result = {**STATUS, "passed": False, "error": str(error)}
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_report(args.output, serialized)
    print(serialized, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
