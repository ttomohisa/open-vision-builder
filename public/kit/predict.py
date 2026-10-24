#!/usr/bin/env python3
"""Run an Open Vision Builder classifier with NumPy, ONNX Runtime, and Pillow.

python predict.py --model model.onnx --image image.png --labels labels.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


def preprocess_image(path: Path, size: int = SIZE) -> np.ndarray:
    """Decode, apply EXIF orientation, float-composite on white, and resize."""
    with Image.open(path) as image:
        rgba = np.asarray(ImageOps.exif_transpose(image).convert("RGBA"), dtype=np.float64)
    height, width = rgba.shape[:2]
    if not isinstance(size, int) or size <= 0 or size > 2048:
        raise ValueError("Target size must be an integer between 1 and 2048")
    alpha = rgba[:, :, 3:4] / 255
    rgb = rgba[:, :, :3] * alpha + 255 * (1 - alpha)
    output = np.empty((1, 3, size, size), dtype=np.float32)
    for oy in range(size):
        sy = (oy + 0.5) * height / size - 0.5
        floor_y = math.floor(sy)
        fy = sy - floor_y
        y0, y1 = max(0, min(height - 1, floor_y)), max(0, min(height - 1, floor_y + 1))
        for ox in range(size):
            sx = (ox + 0.5) * width / size - 0.5
            floor_x = math.floor(sx)
            fx = sx - floor_x
            x0, x1 = max(0, min(width - 1, floor_x)), max(0, min(width - 1, floor_x + 1))
            for channel in range(3):
                top = rgb[y0, x0, channel] * (1 - fx) + rgb[y0, x1, channel] * fx
                bottom = rgb[y1, x0, channel] * (1 - fx) + rgb[y1, x1, channel] * fx
                value = top * (1 - fy) + bottom * fy
                output[0, channel, oy, ox] = (value / 255 - MEAN[channel]) / STD[channel]
    return output


def predict(model: Path, image: Path, labels_file: Path) -> dict:
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels.json must contain a nonempty JSON array of class names")
    if len(set(labels)) != len(labels):
        raise ValueError("Class names must be unique")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Loading bytes keeps this example independent of any checkpoint or external
    # weights beside the single complete model.onnx file supplied by the user.
    session = ort.InferenceSession(model.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, SIZE, SIZE] or inputs[0].type != "tensor(float)":
        raise ValueError("Expected one FP32 image input with shape [1,3,224,224]")
    outputs = session.get_outputs()
    output_name = "logits" if any(value.name == "logits" for value in outputs) else outputs[0].name if len(outputs) == 1 else None
    if output_name is None:
        raise ValueError("Expected a logits output for an Open Vision Builder classifier")
    values = session.run([output_name], {inputs[0].name: preprocess_image(image)})[0]
    if values.shape != (1, len(labels)) or not np.isfinite(values).all():
        raise ValueError("Logits must be finite and match the number of supplied class names")
    logits = values[0].astype(np.float64)
    exponential = np.exp(logits - logits.max())
    probabilities = exponential / exponential.sum()
    order = np.argsort(-probabilities)
    return {
        "predictedLabel": labels[int(order[0])],
        "labels": labels,
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "topPredictions": [
            {"label": labels[int(index)], "probability": float(probabilities[index])}
            for index in order[:5]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = predict(args.model, args.image, args.labels)
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
