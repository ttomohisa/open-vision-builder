#!/usr/bin/env python3
"""Check a complete browser-exported detector and its fixed public parity report.

python scripts/verify_detector.py --model model.onnx --report report.json
Developer verification only: local NumPy/ONNX/ONNX Runtime CPU; no source
checkpoint, model download, user image, browser state, or conversion step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

INPUT_SHAPE = [1, 3, 640, 640]
OUTPUT_SHAPES = {"logits": [1, 300, 80], "boxes": [1, 300, 4]}
TOLERANCES = {"logits": {"atol": 2e-4, "rtol": 1e-4}, "boxes": {"atol": 2e-5, "rtol": 1e-4}}


def fixture_input() -> np.ndarray:
    """Reconstruct dfine-rgb-pattern-v1, phase 0, independently of preparation."""
    y, x = np.indices((640, 640), dtype=np.int64)
    channels = np.stack(((7 * x + 3 * y) % 256, (2 * x + 11 * y) % 256,
                         (13 * x + 5 * y) % 256), axis=0).astype(np.float32)
    return np.ascontiguousarray(channels[None] / np.float32(255))


def tensor_sha256(values: np.ndarray) -> str:
    # Browser FP32 tensor bytes and ONNX raw_data use little-endian storage.
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def expected_tensor(fixture: dict, key: str, shape: list[int]) -> np.ndarray:
    values = fixture.get(key)
    if (not isinstance(values, list) or len(values) != math.prod(shape)
            or any(isinstance(value, bool) or not isinstance(value, (float, int))
                   or not math.isfinite(value) for value in values)):
        raise ValueError(f"{key} must be a flattened finite numeric tensor of shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float32).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"{key} must remain finite in FP32")
    return result


def require_embedded_tensors(message) -> None:
    """Inspect initializers, tensor attributes, nested graphs, and functions."""
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
            raise ValueError("The model must embed every tensor; external data is not allowed")
    for field, value in message.ListFields():
        if field.message_type is not None:
            if field.is_repeated:
                for item in value:
                    require_embedded_tensors(item)
            else:
                require_embedded_tensors(value)


def require_tensor_contract(value, name: str, shape: list[int]) -> None:
    tensor = value.type.tensor_type
    if (value.name != name or tensor.elem_type != onnx.TensorProto.FLOAT
            or [dimension.dim_value for dimension in tensor.shape.dim] != shape):
        raise ValueError(f"Expected FP32 {name} with fixed shape {shape}")


def verify(model_path: Path, report_path: Path) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or type(report.get("schemaVersion")) is not int or report["schemaVersion"] != 1:
        raise ValueError("Expected detector report schemaVersion 1")
    artifact, fixture = report.get("artifact"), report.get("parityFixture")
    if not isinstance(artifact, dict) or artifact.get("file") != "model.onnx":
        raise ValueError("Report must identify the complete model.onnx artifact")
    if (not isinstance(fixture, dict) or fixture.get("name") != "dfine-rgb-pattern-v1"
            or type(fixture.get("phase")) is not int or fixture["phase"] != 0
            or fixture.get("shape") != INPUT_SHAPE):
        raise ValueError("Expected dfine-rgb-pattern-v1 phase 0 with shape [1,3,640,640]")
    verification = report.get("verification")
    if not isinstance(verification, dict) or verification.get("tolerances") != TOLERANCES:
        raise ValueError("Report must use the frozen detector logits/boxes tolerances")

    model_bytes = Path(model_path).read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if artifact.get("sha256") != model_sha256:
        raise ValueError("Model SHA-256 does not match the browser report")
    if type(artifact.get("bytes")) is not int or artifact["bytes"] != len(model_bytes):
        raise ValueError("Model byte length does not match the browser report")
    images = fixture_input()
    input_sha256 = tensor_sha256(images)
    if fixture.get("inputSha256") != input_sha256:
        raise ValueError("Reconstructed public fixture SHA-256 does not match the report")
    expected = {"logits": expected_tensor(fixture, "expectedLogits", OUTPUT_SHAPES["logits"]),
                "boxes": expected_tensor(fixture, "expectedBoxes", OUTPUT_SHAPES["boxes"])}

    # Parsing bytes never follows an external_data location. Reject all such
    # references before invoking the checker or runtime.
    model = onnx.load_model_from_string(model_bytes)
    require_embedded_tensors(model)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1:
        raise ValueError("The complete detector must have only one image input")
    require_tensor_contract(model.graph.input[0], "images", INPUT_SHAPE)
    if len(model.graph.output) != 2 or {value.name for value in model.graph.output} != set(OUTPUT_SHAPES):
        raise ValueError("The complete detector must have exactly logits and boxes outputs")
    for output in model.graph.output:
        require_tensor_contract(output, output.name, OUTPUT_SHAPES[output.name])

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    actual_outputs = session.run(["logits", "boxes"], {"images": images})
    comparison, output_hashes = {}, {}
    for name, actual in zip(("logits", "boxes"), actual_outputs):
        if list(actual.shape) != OUTPUT_SHAPES[name] or actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise ValueError(f"Invalid CPU runtime {name} output")
        reference = expected[name].astype(np.float64)
        errors = np.abs(actual.astype(np.float64) - reference)
        limits = TOLERANCES[name]
        allowance = limits["atol"] + limits["rtol"] * np.abs(reference)
        comparison[name] = {
            "passed": bool(np.all(errors <= allowance)), "shape": OUTPUT_SHAPES[name],
            "maxAbsoluteError": float(errors.max(initial=0)),
            "maxToleranceRatio": float((errors / allowance).max(initial=0)),
            "mismatchedValues": int(np.count_nonzero(errors > allowance)), "values": int(actual.size),
        }
        output_hashes[name] = tensor_sha256(actual)
    return {
        "schemaVersion": 1, "passed": all(value["passed"] for value in comparison.values()),
        "scope": "Independent CPU inference parity on the fixed public FP32 fixture",
        "modelSha256": model_sha256, "modelBytes": len(model_bytes),
        "inputName": "images", "inputShape": INPUT_SHAPE, "inputSha256": input_sha256,
        "fixture": {"name": "dfine-rgb-pattern-v1", "phase": 0},
        "executionProvider": "CPUExecutionProvider", "onnxruntimeVersion": ort.__version__,
        "onnxVersion": onnx.__version__, "numpyVersion": np.__version__,
        "outputSha256": output_hashes, "tolerances": TOLERANCES, "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.model, args.report)
    except Exception as error:
        result = {"schemaVersion": 1, "passed": False, "error": str(error)}
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
