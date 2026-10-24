#!/usr/bin/env python3
"""Verify a browser-exported full ONNX against its saved input/logit fixture.

Example:
    python scripts/predict.py --model model.onnx --verify-report report.json

This helper performs local CPU inference only. It does not access the source
checkpoint, browser caches, or the network, and does not modify the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ATOL = 2e-4
RTOL = 1e-4
INPUT_SHAPE = [1, 3, 224, 224]


def verify(model_path: Path, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixture = report["parityFixture"]
    labels = report["labels"]
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Report labels must be a nonempty list of class names")
    if len(set(labels)) != len(labels):
        raise ValueError("Report class names must be unique")
    if fixture.get("shape") != INPUT_SHAPE:
        raise ValueError(f"Expected fixture input shape {INPUT_SHAPE}")
    values = np.asarray(fixture["input"], dtype=np.float32)
    expected = np.asarray(fixture["expectedLogits"], dtype=np.float32)
    if values.ndim != 1 or values.size != 3 * 224 * 224:
        raise ValueError("Fixture input must be a flattened NCHW tensor")
    if expected.ndim != 1 or expected.size != len(labels):
        raise ValueError("Expected logits must contain one value for each class label")
    if not np.isfinite(values).all() or not np.isfinite(expected).all():
        raise ValueError("Fixture input and expected logits must contain only finite numbers")

    # Read without following any external_data references; reject them explicitly.
    model_bytes = model_path.read_bytes()
    model = onnx.load_model_from_string(model_bytes)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
           for tensor in model.graph.initializer):
        raise ValueError("The deployment model must embed all weights in one ONNX file")
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("The exported classifier must have one image input and one logits output")
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("The deployment model must accept FP32 images")
    if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("The deployment model must produce FP32 logits")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
    image_input = session.get_inputs()[0]
    if image_input.shape != INPUT_SHAPE:
        raise ValueError(f"Unexpected ONNX image input shape {image_input.shape}")
    actual = session.run(None, {image_input.name: values.reshape(INPUT_SHAPE)})[0]
    if actual.shape != (1, len(labels)) or not np.isfinite(actual).all():
        raise ValueError("ONNX output shape or values do not match the report's class list")
    logits = actual[0].astype(np.float64)
    reference = expected.astype(np.float64)
    error = np.abs(logits - reference)
    allowance = ATOL + RTOL * np.abs(reference)
    passed = bool(np.all(error <= allowance))
    return {
        "passed": passed,
        "modelSha256": hashlib.sha256(model_bytes).hexdigest(),
        "executionProvider": "CPUExecutionProvider",
        "onnxruntimeVersion": ort.__version__,
        "inputName": image_input.name,
        "inputShape": INPUT_SHAPE,
        "outputName": session.get_outputs()[0].name,
        "labels": labels,
        "logits": logits.tolist(),
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "comparison": {
            "maxAbsoluteError": float(error.max(initial=0)),
            "maxToleranceRatio": float((error / allowance).max(initial=0)),
            "mismatchedValues": int(np.count_nonzero(error > allowance)),
            "values": len(labels),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verify-report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.model, args.verify_report)
    except Exception as error:
        result = {"passed": False, "error": str(error)}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
