#!/usr/bin/env python3
"""Prepare the pinned public MobileNetV4 checkpoint for the browser P1 slice.

This is a developer/CI task. It accepts no user training images or checkpoints.
The browser app does all user training and final ONNX export locally.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "timm/mobilenetv4_conv_small.e2400_r224_in1k"
SOURCE_REVISION = "331fb803779522b685cf942e15f914fb6741c1eb"
SOURCE_SHA256 = "7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4"
SOURCE_BYTES = 15_223_016
TIMM_COMMIT = "e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d"
SOURCE_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{SOURCE_REVISION}"
SOURCE_URL = f"{SOURCE_BASE}/model.safetensors"
TIMM_BASE = f"https://raw.githubusercontent.com/huggingface/pytorch-image-models/{TIMM_COMMIT}"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224
FEATURE_SIZE = 1280
ATOL = 2e-4
RTOL = 1e-4
FIXTURE_NAME = "rgba-pattern-v1"
RESIZE_RECIPE = "bilinear-half-pixel-stretch"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_public(url: str, target: Path, expected_hash: str | None = None) -> Path:
    """Cache only immutable public sources; never load an unverified checkpoint."""
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with urlopen(url, timeout=60) as response, partial.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if expected_hash and sha256_file(partial) != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {url}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    if expected_hash and sha256_file(target) != expected_hash:
        raise ValueError(f"Cached public checkpoint SHA-256 mismatch: {target}")
    return target


def fixture_rgba() -> np.ndarray:
    """A generated, non-photographic 40 x 32 fixture shared with the JS runner."""
    y, x = np.indices((32, 40), dtype=np.int64)
    return np.stack(
        ((x * 7 + y * 3) % 256, (x * 2 + y * 11) % 256,
         (x * 13 + y * 5) % 256, np.full_like(x, 255)), axis=-1,
    ).astype(np.uint8)


def preprocess_rgba(rgba, width: int, height: int, size: int = SIZE) -> np.ndarray:
    """RGB half-pixel bilinear stretch, float64 arithmetic, final NCHW float32.

    source_coordinate = (destination + .5) * source_size / target_size - .5.
    Both source indices are clamped independently. RGBA is composited on white
    with floating-point alpha before resizing. There is no intermediate pixel
    rounding, gamma conversion, or crop.
    """
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (width, height, size)):
        raise ValueError("Image dimensions must be positive integers")
    pixels = np.asarray(rgba, dtype=np.float64)
    if pixels.size != width * height * 4:
        raise ValueError("RGBA data does not match the declared image dimensions")
    if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 255).any():
        raise ValueError("RGBA values must be finite numbers between 0 and 255")
    pixels = pixels.reshape(height, width, 4)
    alpha = pixels[:, :, 3:4] / 255
    rgb = pixels[:, :, :3] * alpha + 255 * (1 - alpha)
    output = np.empty((1, 3, size, size), dtype=np.float32)
    for oy in range(size):
        source_y = (oy + 0.5) * height / size - 0.5
        floor_y = math.floor(source_y)
        fy = source_y - floor_y
        y0, y1 = max(0, min(height - 1, floor_y)), max(0, min(height - 1, floor_y + 1))
        for ox in range(size):
            source_x = (ox + 0.5) * width / size - 0.5
            floor_x = math.floor(source_x)
            fx = source_x - floor_x
            x0, x1 = max(0, min(width - 1, floor_x)), max(0, min(width - 1, floor_x + 1))
            for channel in range(3):
                top = rgb[y0, x0, channel] * (1 - fx) + rgb[y0, x1, channel] * fx
                bottom = rgb[y1, x0, channel] * (1 - fx) + rgb[y1, x1, channel] * fx
                value = top * (1 - fy) + bottom * fy
                output[0, channel, oy, ox] = (value / 255 - MEAN[channel]) / STD[channel]
    return output


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Parity tensors must have the same shape and only finite values")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowance = ATOL + RTOL * np.abs(expected.astype(np.float64))
    result = {
        "passed": bool(np.all(error <= allowance)),
        "maxAbsoluteError": float(error.max(initial=0)),
        "maxToleranceRatio": float((error / allowance).max(initial=0)),
        "mismatchedValues": int(np.count_nonzero(error > allowance)),
        "values": int(error.size),
    }
    if not result["passed"]:
        raise ValueError(f"Native/ONNX parity failed: {json.dumps(result)}")
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(output: Path, cache: Path) -> dict:
    import onnx
    import onnxruntime as ort
    import torch
    import timm
    from safetensors.torch import load_file

    required_versions = {
        "torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu", "timm": "1.0.15",
        "onnx": "1.17.0", "onnxruntime": "1.20.1", "numpy": "2.2.3",
        "safetensors": "0.5.3",
    }
    for package, required in required_versions.items():
        found = importlib.metadata.version(package)
        if found != required:
            raise RuntimeError(f"Expected {package}=={required}, found {found}; use requirements-models.txt")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checkpoint = download_public(SOURCE_URL, cache / "model.safetensors", SOURCE_SHA256)
    if checkpoint.stat().st_size != SOURCE_BYTES:
        raise ValueError("Checkpoint byte length does not match the pinned public source")
    source_files = {
        "MODEL_CARD.md": download_public(f"{SOURCE_BASE}/README.md", cache / "MODEL_CARD.md",
                                         "864651a7a9c0457023728da202cb1119abf3fa0872ded374ba736efde6454eb7"),
        "SOURCE_CONFIG.json": download_public(f"{SOURCE_BASE}/config.json", cache / "SOURCE_CONFIG.json",
                                              "ecd20bcf1287aaa88129a736d188904d0b134c96e152b55b241be895657fe41a"),
        "LICENSE.txt": download_public(f"{TIMM_BASE}/LICENSE", cache / "LICENSE.txt",
                                       "71b111620fa32c17a80ccd6db2da759d31a591e497d8b5f382174f59d1b59d49"),
    }
    card = source_files["MODEL_CARD.md"].read_text(encoding="utf-8")
    if "license: apache-2.0" not in card.split("---", 2)[1]:
        raise ValueError("The pinned source model card does not declare Apache-2.0")
    source_config = json.loads(source_files["SOURCE_CONFIG.json"].read_text(encoding="utf-8"))
    if source_config.get("architecture") != "mobilenetv4_conv_small" or source_config.get("num_classes") != 1000:
        raise ValueError("Unexpected pinned source architecture")

    state = load_file(str(checkpoint), device="cpu")
    if tuple(state["classifier.weight"].shape) != (1000, FEATURE_SIZE):
        raise ValueError("Unexpected native classifier shape")
    if tuple(state["conv_head.weight"].shape) != (FEATURE_SIZE, 960, 1, 1):
        raise ValueError("Unexpected native conv_head shape")
    if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise ValueError("Source checkpoint contains non-finite tensors")
    base = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False)
    base.load_state_dict(state, strict=True)
    base.requires_grad_(False).eval().float()

    class ExportModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.base = model

        def forward(self, images):
            features = self.base.forward_head(self.base.forward_features(images), pre_logits=True)
            return self.base.classifier(features), features

    wrapper = ExportModel(base).eval()
    input_values = preprocess_rgba(fixture_rgba(), 40, 32)
    images = torch.from_numpy(input_values)
    with torch.inference_mode():
        native_logits, native_features = (value.numpy().copy() for value in wrapper(images))
    if native_logits.shape != (1, 1000) or native_features.shape != (1, FEATURE_SIZE):
        raise ValueError("Unexpected native output shape")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as staging_name:
        staging = Path(staging_name)
        model_file = staging / "backbone.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, images, str(model_file), input_names=["images"],
                output_names=["logits", "features"], opset_version=17, dynamo=False,
                do_constant_folding=True, keep_initializers_as_inputs=False,
                export_params=True, external_data=False,
            )
        model = onnx.load(str(model_file), load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        if any(tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
               for tensor in initializers.values()):
            raise ValueError("External ONNX weights are not allowed")
        if any(tensor.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE, onnx.TensorProto.BFLOAT16)
               for tensor in initializers.values()):
            raise ValueError("All floating-point inference weights must be FP32")
        shape = lambda value: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "images" or shape(model.graph.input[0]) != [1, 3, SIZE, SIZE]:
            raise ValueError("Unexpected ONNX input contract")
        outputs = {value.name: shape(value) for value in model.graph.output}
        if outputs != {"logits": [1, 1000], "features": [1, FEATURE_SIZE]}:
            raise ValueError(f"Unexpected ONNX output contract: {outputs}")
        heads = [node for node in model.graph.node if "logits" in node.output]
        if len(heads) != 1 or heads[0].op_type != "Gemm" or len(heads[0].input) != 3:
            raise ValueError("Expected exactly one native final Gemm classifier")
        head = heads[0]
        attrs = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in head.attribute}
        if head.input[0] != "features" or attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1:
            raise ValueError("Unexpected classifier feature wiring or transpose convention")
        if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
            raise ValueError("Unexpected classifier scaling")
        if list(initializers[head.input[1]].dims) != [1000, FEATURE_SIZE] or list(initializers[head.input[2]].dims) != [1000]:
            raise ValueError("Unexpected classifier initializer shapes")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_file), sess_options=session_options, providers=["CPUExecutionProvider"])
        actual_logits, actual_features = session.run(["logits", "features"], {"images": input_values})
        parity = {"logits": comparison(actual_logits, native_logits), "features": comparison(actual_features, native_features)}
        tool_versions = {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "timm", "onnx", "onnxruntime", "numpy", "safetensors", "Pillow", "protobuf")
        }}
        reference = {
            "schemaVersion": 1,
            "fixture": {"name": FIXTURE_NAME, "width": 40, "height": 32},
            "native": {"features": native_features.ravel().tolist(), "logits": native_logits.ravel().tolist()},
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "toolVersions": tool_versions,
        }
        manifest = {
            "schemaVersion": 1,
            "id": "mobilenetv4-small",
            "displayName": "MobileNetV4 Conv Small · ImageNet-1k",
            "source": {"modelId": MODEL_ID, "revision": SOURCE_REVISION, "file": "model.safetensors",
                       "sha256": SOURCE_SHA256, "license": "Apache-2.0", "url": SOURCE_URL},
            "artifact": {"file": "backbone.onnx", "sha256": sha256_file(model_file), "bytes": model_file.stat().st_size},
            "input": {"name": "images", "shape": [1, 3, SIZE, SIZE], "mean": list(MEAN), "std": list(STD), "resize": RESIZE_RECIPE},
            "features": {"name": "features", "size": FEATURE_SIZE},
            "originalOutput": "logits",
            "head": {"nodeName": head.name, "weightName": head.input[1], "biasName": head.input[2]},
            "reference": {"file": "reference.json", "fixture": FIXTURE_NAME},
            "licenses": {"licenseFile": "LICENSE.txt", "noticeFile": "NOTICE.txt", "modelCardFile": "MODEL_CARD.md"},
        }
        notice = f"""Open Vision Builder — public base-model attribution and modification record

Model: {MODEL_ID}
Source revision: {SOURCE_REVISION}
Source file: model.safetensors
Source SHA-256: {SOURCE_SHA256}
Source URL: {SOURCE_URL}

The source model card declares Apache-2.0. It identifies Ross Wightman as
the trainer and ImageNet-1k as the training dataset. MODEL_CARD.md is the
unmodified model card from this fixed revision. LICENSE.txt is the Apache
License 2.0 from timm commit {TIMM_COMMIT}.
timm implementation: https://github.com/huggingface/pytorch-image-models
MobileNetV4 architecture: https://arxiv.org/abs/2404.10518

Changes made by Open Vision Builder:
- Converted the complete FP32 eval model to a single embedded-weight ONNX,
  opset 17, using torch 2.6.0 and timm 1.0.15.
- Exposed the native 1280-dimensional pre-logit tensor as an additional output.
- Allowed the exporter to fold eval BatchNorm into convolution weights;
  all native trainable parameters were loaded strictly before conversion.
- Defined floating-point RGBA compositing on white, RGB full-image bilinear
  half-pixel stretch to 224 x 224, followed by ImageNet normalization. No
  intermediate 8-bit rounding is applied. This differs from the source model's bicubic
  resize and center-crop recipe. Upstream accuracy figures do not establish
  the accuracy of this application recipe.

The browser's derived model keeps the frozen pretrained feature extractor
and replaces the 1000-class classifier with a head trained locally by the
user. Preserve this attribution, LICENSE.txt, and MODEL_CARD.md with it,
and record that subsequent change in the exported model/report.

This attribution file was written downstream; no upstream root NOTICE file
was present in the inspected timm commit. The fixture is generated from
integer pixel formulas and contains no user data or third-party photograph.
"""
        for name, source_path in source_files.items():
            (staging / name).write_bytes(source_path.read_bytes())
        (staging / "NOTICE.txt").write_text(notice, encoding="utf-8")
        evidence = {
            "status": "native-to-onnx-verified",
            "scope": "Fixed synthetic fixture on Python CPU; browser training and real-world quality are not established here.",
            "source": manifest["source"],
            "code": {"timmVersion": "1.0.15", "timmCommit": TIMM_COMMIT, "evalMode": True, "strictLoad": True},
            "artifact": manifest["artifact"],
            "inputSha256": hashlib.sha256(input_values.astype("<f4", copy=False).tobytes()).hexdigest(),
            "recipe": {"name": RESIZE_RECIPE, "sourceCoordinate": "(destination + 0.5) * sourceSize / targetSize - 0.5",
                       "clamp": "source indices independently", "intermediatePixelRounding": False,
                       "alpha": "float RGB * (alpha/255) + 255 * (1-alpha/255), before resize",
                       "sourceRecipe": source_config["pretrained_cfg"], "inheritsUpstreamAccuracy": False},
            "fixture": {"name": FIXTURE_NAME, "width": 40, "height": 32,
                        "r": "(x*7+y*3)%256", "g": "(x*2+y*11)%256", "b": "(x*13+y*5)%256", "a": 255},
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "comparison": parity,
            "graph": {"irVersion": model.ir_version, "opset": [{"domain": entry.domain, "version": entry.version} for entry in model.opset_import],
                      "nodes": len(model.graph.node), "initializers": len(initializers),
                      "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
                      "head": {"name": head.name, "inputs": list(head.input), "attributes": attrs},
                      "externalWeights": False,
                      "nativeClassifierShape": list(state["classifier.weight"].shape),
                      "nativeConvHeadShape": list(state["conv_head.weight"].shape)},
            "environment": {"platform": platform.platform(), "toolVersions": tool_versions, "executionProvider": "CPUExecutionProvider", "threads": 1},
            "sourceDocumentSha256": {name: sha256_file(path) for name, path in source_files.items()},
        }
        write_json(staging / "reference.json", reference)
        write_json(staging / "preparation.json", evidence)
        write_json(staging / "model.json", manifest)
        # Publish the manifest last, after every check passes and every payload exists.
        for path in staging.iterdir():
            if path.name != "model.json":
                path.replace(output / path.name)
        (staging / "model.json").replace(output / "model.json")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "public/models/mobilenetv4-small")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".venv/model-source-cache")
    args = parser.parse_args()
    try:
        result = prepare(args.output_dir, args.cache_dir)
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, "output": str(args.output_dir), "artifact": result["artifact"], "comparison": result["comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
