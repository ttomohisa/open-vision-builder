#!/usr/bin/env python3
"""Run a complete Open Vision Builder D-FINE ONNX with local inference only.

python detect.py --model model.onnx --image image.png --labels labels.json
Dependencies: numpy, Pillow, onnxruntime. No model download or conversion.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

SIZE = 640
QUERIES = 300
CLASSES = 80
INPUT_SHAPE = [1, 3, SIZE, SIZE]
OUTPUT_SHAPES = {"logits": [1, QUERIES, CLASSES], "boxes": [1, QUERIES, 4]}


def preprocess_rgba(rgba: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Float-composite RGBA on white, stretch with half-pixel bilinear, RGB/255."""
    pixels = np.asarray(rgba)
    if pixels.ndim != 3 or pixels.shape[2] != 4 or min(pixels.shape[:2]) <= 0:
        raise ValueError("Expected nonempty RGBA pixels with shape [height,width,4]")
    if pixels.dtype != np.uint8:
        raise ValueError("Decoded RGBA pixels must contain uint8 channel values")
    if type(size) is not int or not 1 <= size <= 2048:
        raise ValueError("Target size must be an integer between 1 and 2048")
    height, width = pixels.shape[:2]
    pixels = pixels.astype(np.float64)
    alpha = pixels[:, :, 3:4] / 255
    rgb = pixels[:, :, :3] * alpha + 255 * (1 - alpha)

    # Neighbor coordinates come from the raw floor before edge clipping.
    x = (np.arange(size, dtype=np.float64) + 0.5) * width / size - 0.5
    y = (np.arange(size, dtype=np.float64) + 0.5) * height / size - 0.5
    floor_x, floor_y = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - floor_x, y - floor_y
    x0, x1 = np.clip(floor_x, 0, width - 1), np.clip(floor_x + 1, 0, width - 1)
    y0, y1 = np.clip(floor_y, 0, height - 1), np.clip(floor_y + 1, 0, height - 1)
    top = (rgb[y0[:, None], x0[None, :]] * (1 - fx)[None, :, None]
           + rgb[y0[:, None], x1[None, :]] * fx[None, :, None])
    bottom = (rgb[y1[:, None], x0[None, :]] * (1 - fx)[None, :, None]
              + rgb[y1[:, None], x1[None, :]] * fx[None, :, None])
    resized = top * (1 - fy)[:, None, None] + bottom * fy[:, None, None]
    return np.ascontiguousarray((resized / 255).transpose(2, 0, 1)[None], dtype=np.float32)


def preprocess_image(path: Path, size: int = SIZE) -> tuple[np.ndarray, tuple[int, int]]:
    """Return FP32 input plus the EXIF-oriented original (width,height)."""
    with Image.open(path) as image:
        # ImageCms is intentionally not used: embedded ICC profiles do not
        # change the decoded channel values in this preprocessing contract.
        pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGBA"), dtype=np.uint8)
    height, width = pixels.shape[:2]
    return preprocess_rgba(pixels, size), (width, height)


def validate_labels(labels: list[str]) -> None:
    if not isinstance(labels, list) or len(labels) != CLASSES or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("labels.json must contain exactly 80 nonempty class names in model-output order")


def postprocess_predictions(logits: np.ndarray, boxes: np.ndarray, labels: list[str],
                            image_size: tuple[int, int], threshold: float = 0.3,
                            max_detections: int = 100) -> list[dict]:
    """Global sigmoid query/class ranking; original-pixel xyxy, no NMS or clipping."""
    validate_labels(labels)
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)):
        raise ValueError("Original image size must contain positive integer width and height")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Score threshold must be a finite number between 0 and 1")
    if type(max_detections) is not int or not 1 <= max_detections <= QUERIES * CLASSES:
        raise ValueError("max_detections must be an integer between 1 and 24000")
    logits, boxes = np.asarray(logits), np.asarray(boxes)
    for name, array in (("logits", logits), ("boxes", boxes)):
        if list(array.shape) != OUTPUT_SHAPES[name] or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"Expected finite floating-point {name} with shape {OUTPUT_SHAPES[name]}")
    if np.any(boxes[..., 2:] < 0):
        raise ValueError("Box widths and heights must not be negative")

    flat = logits.reshape(-1).astype(np.float64)
    scores = np.empty_like(flat)
    positive = flat >= 0
    scores[positive] = 1 / (1 + np.exp(-flat[positive]))
    negative_exp = np.exp(flat[~positive])
    scores[~positive] = negative_exp / (1 + negative_exp)
    # Stable sorting resolves equal scores by flattened query/class index.
    selected = np.argsort(-scores, kind="stable")[:max_detections]
    selected = selected[scores[selected] >= threshold]
    width, height = image_size
    detections = []
    for index in selected:
        query, class_index = divmod(int(index), CLASSES)
        cx, cy, box_width, box_height = boxes[0, query].astype(np.float64)
        detections.append({
            "queryIndex": query, "classIndex": class_index, "label": labels[class_index],
            "score": float(scores[index]),
            "box": [float((cx - box_width / 2) * width), float((cy - box_height / 2) * height),
                    float((cx + box_width / 2) * width), float((cy + box_height / 2) * height)],
        })
    return detections


def detect(model: Path, image: Path, labels_file: Path, threshold: float = 0.3,
           max_detections: int = 100) -> dict:
    labels = json.loads(Path(labels_file).read_text(encoding="utf-8"))
    validate_labels(labels)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # Loading bytes prevents resolving external weights relative to a model
    # path. The exported model must be independently usable as one ONNX file.
    session = ort.InferenceSession(Path(model).read_bytes(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "images" or inputs[0].shape != INPUT_SHAPE
            or inputs[0].type != "tensor(float)"):
        raise ValueError("Expected one FP32 images input with shape [1,3,640,640]")
    if len(outputs) != 2 or {value.name for value in outputs} != set(OUTPUT_SHAPES):
        raise ValueError("Expected exactly logits and boxes outputs")
    for output in outputs:
        if output.type != "tensor(float)" or output.shape != OUTPUT_SHAPES[output.name]:
            raise ValueError(f"Unexpected ONNX {output.name} output type or shape")
    tensor, image_size = preprocess_image(Path(image))
    logits, boxes = session.run(["logits", "boxes"], {"images": tensor})
    return {
        "image": {"width": image_size[0], "height": image_size[1]},
        "boxFormat": "xyxy", "coordinateSpace": "original-oriented-image-pixels",
        "scoreThreshold": threshold, "maxDetections": max_detections,
        "postprocessing": {"activation": "sigmoid", "ranking": "global-query-class", "nms": False, "clipBoxes": False},
        "detections": postprocess_predictions(logits, boxes, labels, image_size, threshold, max_detections),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = detect(args.model, args.image, args.labels, args.threshold, args.max_detections)
        status = 0
    except Exception as error:
        result, status = {"error": str(error)}, 1
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
