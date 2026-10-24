#!/usr/bin/env python3
"""Prepare the pinned public COCO-only D-FINE Nano investigation assets.

Developer/CI only: no user checkpoint, image, label, runtime server, or pickle
loader is accepted. The default output is ignored evidence, not an app asset.
The reference is the pinned Transformers PyTorch model; equality with a native
D-FINE .pth checkpoint and full detector training are separate, unproved gates.
"""

from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "ustc-community/dfine-nano-coco"
SOURCE_REVISION = "066438d3d8f0da137a37b38fdf3368fd4afceced"
SOURCE_SHA256 = "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"
SOURCE_BYTES = 15_278_996
SOURCE_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{SOURCE_REVISION}"
SOURCE_URL = f"{SOURCE_BASE}/model.safetensors"
TRANSFORMERS_COMMIT = "e7d351cebad5f6dcdd169b0c034fdee0a000e6a9"
TRANSFORMERS_BASE = f"https://raw.githubusercontent.com/huggingface/transformers/{TRANSFORMERS_COMMIT}"
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
DFINE_BASE = f"https://raw.githubusercontent.com/Peterande/D-FINE/{DFINE_COMMIT}"
CONVERSION_COMMIT = "be5169c704bae3d1d76d2890be5c72bcbc8c5de2"
SIZE = 640
OUTPUT_SHAPES = {
    "logits": [1, 300, 80], "boxes": [1, 300, 4],
    "bboxInput": [1, 300, 128], "classInput": [1, 300, 128],
    "residualCorners": [1, 300, 132], "references": [1, 300, 4],
}
# Declared before the first parity measurement; apply to every finite element.
TOLERANCES = {name: {"atol": 2e-5 if name in ("boxes", "references") else 2e-4, "rtol": 1e-4}
              for name in OUTPUT_SHAPES}
FIXTURE_PHASES = (0, 37, 91)
DEFAULT_DIAGNOSTIC_DIR = ROOT / "artifacts/p1b/detector-assets/unqualified"
FAILED_STATUS = "unqualified-intermediate-parity-failure"
REQUIRED_VERSIONS = {
    "torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu", "transformers": "4.56.0",
    "onnx": "1.17.0", "onnxruntime": "1.20.1", "numpy": "2.2.3",
    "safetensors": "0.5.3", "Pillow": "11.1.0", "protobuf": "7.36.2",
    "huggingface-hub": "0.34.4", "tokenizers": "0.22.0",
}
DOCUMENTS = {
    "MODEL_CARD.md": (f"{SOURCE_BASE}/README.md", "49aff1e9de08165807d9581bc728f7d1a0f7944ba8a1913c3a2251e24cfbfaf9", 2809),
    "SOURCE_CONFIG.json": (f"{SOURCE_BASE}/config.json", "7edf2d96d3deb10aa4ec920be0e8e444b617454545bb58093e32bd8e2b96fa2f", 6557),
    "PREPROCESSOR_CONFIG.json": (f"{SOURCE_BASE}/preprocessor_config.json", "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1", 444),
    "LICENSE.txt": (f"{DFINE_BASE}/LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11357),
    "D_FINE_README.md": (f"{DFINE_BASE}/README.md", "d723d80814f51790fde117a021de6a04e5ab2884d5992a46b8605a6578fbb218", 27586),
    "TRANSFORMERS_LICENSE.txt": (f"{TRANSFORMERS_BASE}/LICENSE", "77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049", 11418),
    "modeling_d_fine.py": (f"{TRANSFORMERS_BASE}/src/transformers/models/d_fine/modeling_d_fine.py", "f0d05693550a30d0fe049d9c5bdd77f4363df5be06c8ab3cf6f7bd07fd6a74f0", 105305),
    "configuration_d_fine.py": (f"{TRANSFORMERS_BASE}/src/transformers/models/d_fine/configuration_d_fine.py", "98583fc5c0f6b7bf9ebe26d7e5110a7822a0dd129921d050b1246d1969d837a1", 22689),
    "CONVERSION.py": (f"https://raw.githubusercontent.com/huggingface/transformers/{CONVERSION_COMMIT}/src/transformers/models/d_fine/convert_d_fine_original_pytorch_checkpoint_to_hf.py", "b6578386bd5d5dd6cb10f015bbb41cb2360ad83377e3e046e8d8d777ac60a7fb", 38785),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pinned(url, target, expected_sha256, expected_bytes):
    """Do not replace an existing corrupt cache entry or load unchecked bytes."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    def verify(path):
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Pinned byte length mismatch: {path}")

    if target.exists():
        verify(target)
        return target
    partial = target.with_name(target.name + ".part")
    try:
        with urlopen(url, timeout=60) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        verify(partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class NumericParityError(ValueError):
    def __init__(self, name, result):
        self.result = result
        super().__init__(f"Native/ONNX parity failed for {name}: {json.dumps(result)}")


class QualificationFailure(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = str(diagnostic)
        super().__init__(f"Detector qualification failed; unqualified evidence: {diagnostic}")


def tensor_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(tensor_array(value)).tobytes()).hexdigest()


def fixture_input(phase=0):
    """Original procedural RGB fixture: no downloaded photograph or user data."""
    if phase not in FIXTURE_PHASES:
        raise ValueError("Unknown fixed fixture phase")
    y, x = np.mgrid[:SIZE, :SIZE]
    rgb = np.stack(((7 * x + 3 * y + phase) % 256,
                    (2 * x + 11 * y + 3 * phase) % 256,
                    (13 * x + 5 * y + 7 * phase) % 256), axis=0).astype(np.float32)
    return np.ascontiguousarray(rgb[None] / np.float32(255))


def comparison(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Invalid parity tensor: {name}")
    tolerance = TOLERANCES[name]
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bound = tolerance["atol"] + tolerance["rtol"] * np.abs(expected.astype(np.float64))
    result = {"passed": bool(np.all(error <= bound)), "shape": list(expected.shape),
              "maxAbsoluteError": float(error.max(initial=0)),
              "maxToleranceRatio": float((error / bound).max(initial=0)), **tolerance}
    if not result["passed"]:
        raise NumericParityError(name, result)
    return result


def bind_output_shapes(model, observed_shapes):
    """Bind unresolved output metadata only to measured fixed-input shapes.

    PyTorch's legacy exporter leaves batch/query symbols after attention and
    Gather even for the fixed input. Never overwrite a contradictory known dim.
    Every annotated dimension is checked again by all ORT fixture executions.
    """
    if {value.name for value in model.graph.output} != set(observed_shapes):
        raise ValueError("Observed and exported output names differ")
    for value in model.graph.output:
        shape = value.type.tensor_type.shape.dim
        observed = observed_shapes[value.name]
        if len(shape) != len(observed):
            raise ValueError("Export rank contradicts observed output shape")
        for dimension, size in zip(shape, observed):
            if dimension.HasField("dim_value") and dimension.dim_value != size:
                raise ValueError("Known export dimension contradicts observed output shape")
            dimension.dim_value = size


def canonicalize_fp32_constants(model):
    """Perform only FP64-scalar -> Cast(FP32) constant evaluation.

    Legacy export emits a double epsilon constant used exclusively by FP32
    casts. Converting that constant produces the same float32 bits as each
    existing cast; no FP64 arithmetic, parameter, or other use is rewritten.
    """
    import onnx
    from onnx import numpy_helper

    changes = []
    outputs = {value.name for value in model.graph.output}
    for node in model.graph.node:
        if node.op_type != "Constant" or node.domain or len(node.output) != 1:
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        attribute = attributes.get("value")
        if attribute is None or attribute.type != onnx.AttributeProto.TENSOR:
            continue
        tensor = attribute.t
        if tensor.data_type != onnx.TensorProto.DOUBLE or tensor.dims or node.output[0] in outputs:
            continue
        consumers = [consumer for consumer in model.graph.node if node.output[0] in consumer.input]
        if not consumers or any(consumer.op_type != "Cast" or consumer.domain or
                                {a.name: onnx.helper.get_attribute_value(a) for a in consumer.attribute}.get("to") != onnx.TensorProto.FLOAT
                                for consumer in consumers):
            continue
        values = numpy_helper.to_array(tensor)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite scalar constant")
        converted = values.astype(np.float32)
        changes.append({"output": node.output[0], "from": "DOUBLE", "to": "FLOAT",
                        "consumers": [consumer.name for consumer in consumers],
                        "originalSha256": tensor_sha256(values), "convertedSha256": tensor_sha256(converted)})
        tensor.CopyFrom(numpy_helper.from_array(converted, tensor.name))
        for value in model.graph.value_info:
            if value.name == node.output[0]:
                value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return changes


def tensor_inventory(model):
    """Audit nested graphs and tensor attributes, not only top-level weights."""
    import onnx

    tensors, nodes, values = [], [], []
    allowed_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.INT64,
                     onnx.TensorProto.INT32, onnx.TensorProto.BOOL}

    def add_tensor(tensor, location):
        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(f"External tensor is not allowed: {location}")
        if tensor.data_type not in allowed_types:
            raise ValueError(f"Non-FP32 floating or unsupported tensor dtype: {location}")
        tensors.append({"name": tensor.name, "location": location, "shape": list(tensor.dims),
                        "dtype": onnx.TensorProto.DataType.Name(tensor.data_type)})

    def visit(graph, path):
        if graph.sparse_initializer:
            raise ValueError("Sparse initializers are not part of this export contract")
        for tensor in graph.initializer:
            add_tensor(tensor, path + "/initializer/" + tensor.name)
        for value in (*graph.input, *graph.output, *graph.value_info):
            if value.type.HasField("tensor_type"):
                dtype = value.type.tensor_type.elem_type
                if dtype and dtype not in allowed_types:
                    raise ValueError(f"Non-FP32 floating or unsupported value dtype: {value.name}")
                values.append({"name": value.name, "dtype": onnx.TensorProto.DataType.Name(dtype),
                               "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param or None
                                         for d in value.type.tensor_type.shape.dim]})
        for index, node in enumerate(graph.node):
            if node.domain:
                raise ValueError(f"Unsupported operator domain: {node.domain}")
            nodes.append(node)
            for attribute in node.attribute:
                location = f"{path}/node{index}/{attribute.name}"
                if attribute.type == onnx.AttributeProto.TENSOR:
                    add_tensor(attribute.t, location)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        add_tensor(tensor, location)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g, location)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_value in attribute.graphs:
                        visit(graph_value, location)

    if model.functions:
        raise ValueError("Local functions require a separate operator audit")
    if any(entry.domain for entry in model.opset_import):
        raise ValueError("Unsupported opset domain")
    visit(model.graph, "graph")
    return {"irVersion": model.ir_version,
            "opsets": [{"domain": entry.domain, "version": entry.version} for entry in model.opset_import],
            "nodes": len(nodes), "initializers": len(model.graph.initializer),
            "operators": dict(sorted(Counter(node.op_type for node in nodes).items())),
            "domains": sorted({node.domain for node in nodes}),
            "tensorDtypes": dict(sorted(Counter(item["dtype"] for item in tensors).items())),
            "valueDtypes": dict(sorted(Counter(item["dtype"] for item in values).items())),
            "externalData": False, "tensors": tensors, "values": values}


def map_parameters(model, state):
    """Source identity and exact bytes for every directly named initializer."""
    from onnx import numpy_helper

    mapped = []
    for initializer in model.graph.initializer:
        source_key = initializer.name.removeprefix("base.")
        if source_key not in state:
            continue
        source, actual = tensor_array(state[source_key]), numpy_helper.to_array(initializer)
        if source.dtype != actual.dtype or source.shape != actual.shape or not np.array_equal(source, actual):
            raise ValueError(f"Source initializer value mismatch: {initializer.name}")
        mapped.append({"sourceKey": source_key, "initializerName": initializer.name,
                       "shape": list(source.shape), "dtype": str(source.dtype),
                       "transform": "identity", "sha256": tensor_sha256(source)})
    return mapped


def load_source_model(cache):
    """Strict safe load, with only the architecture's documented shared heads."""
    import torch
    from safetensors.torch import load_file
    from transformers import DFineConfig, DFineForObjectDetection
    from transformers.models.d_fine import modeling_d_fine, configuration_d_fine

    cache = Path(cache)
    for package, wanted in REQUIRED_VERSIONS.items():
        if importlib.metadata.version(package) != wanted:
            raise RuntimeError(f"Use requirements-detector.txt: expected {package}=={wanted}")
    for module in (modeling_d_fine, configuration_d_fine):
        name = Path(module.__file__).name
        if sha256_file(module.__file__) != DOCUMENTS[name][1]:
            raise ValueError(f"Installed implementation source differs from pinned commit: {name}")
    checkpoint = download_pinned(SOURCE_URL, cache / "model.safetensors", SOURCE_SHA256, SOURCE_BYTES)
    for name, (url, digest, size) in DOCUMENTS.items():
        download_pinned(url, cache / name, digest, size)
    card = (cache / "MODEL_CARD.md").read_text(encoding="utf-8")
    if "license: apache-2.0" not in card.split("---", 2)[1]:
        raise ValueError("Pinned model card does not declare Apache-2.0")
    config_dict = json.loads((cache / "SOURCE_CONFIG.json").read_text())
    preprocessor = json.loads((cache / "PREPROCESSOR_CONFIG.json").read_text())
    if (config_dict["model_type"], config_dict["decoder_layers"], config_dict["d_model"],
        config_dict["num_queries"], len(config_dict["id2label"])) != ("d_fine", 3, 128, 300, 80):
        raise ValueError("Unexpected source model architecture")
    if (preprocessor["size"] != {"height": SIZE, "width": SIZE} or preprocessor["do_normalize"]
            or preprocessor["do_pad"] or preprocessor["rescale_factor"] != 1 / 255):
        raise ValueError("Unexpected source preprocessing recipe")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    state = load_file(str(checkpoint), device="cpu")
    if len(state) != 644:
        raise ValueError("Unexpected number of source tensors")
    if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise ValueError("Source contains non-finite weights")
    for index in range(3):
        prefix = f"model.decoder.layers.{index}.self_attn."
        if torch.equal(state[prefix + "k_proj.weight"], state[prefix + "v_proj.weight"]):
            raise ValueError("Checkpoint has the historical duplicated K/V conversion defect")
    base = DFineForObjectDetection(DFineConfig.from_dict(config_dict))
    expected = base.state_dict()
    aliases = {}
    expanded = dict(state)
    for name in sorted(set(expected) - set(state)):
        source_key = "model.decoder." + name
        if not name.startswith(("bbox_embed.", "class_embed.")) or source_key not in state:
            raise ValueError(f"Unexpected missing source tensor: {name}")
        aliases[name] = source_key
        expanded[name] = state[source_key]
    if len(aliases) != 24 or set(state) - set(expected):
        raise ValueError("Unexpected source aliases or extra tensors")
    base.load_state_dict(expanded, strict=True)
    # Confirm that the extra names really share storage, rather than filling a
    # missing independent parameter with a plausible-looking source tensor.
    loaded = base.state_dict()
    if any(loaded[name].data_ptr() != loaded[source].data_ptr() for name, source in aliases.items()):
        raise ValueError("A declared head alias does not share the same parameter")
    base.requires_grad_(False).eval().float()
    return base, state, aliases


def export_wrapper(base):
    """Expose exact eval tensors without changing the upstream forward code."""
    import torch

    class ExportModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.taps = {}
            for index in (0, 1):
                base.bbox_embed[index].register_forward_hook(
                    lambda _module, _args, output, index=index: self.taps.__setitem__(f"corner{index}", output))
            base.bbox_embed[2].register_forward_pre_hook(
                lambda _module, args: self.taps.__setitem__("bboxInput", args[0]))
            base.class_embed[2].register_forward_pre_hook(
                lambda _module, args: self.taps.__setitem__("classInput", args[0]))

        def forward(self, images):
            self.taps.clear()
            output = self.base(pixel_values=images, return_dict=True)
            return (output.logits, output.pred_boxes, self.taps["bboxInput"], self.taps["classInput"],
                    self.taps["corner0"] + self.taps["corner1"], output.initial_reference_points[:, -1])

    return ExportModel().eval()


def _build_assets(staging, cache):
    import onnx
    import onnxruntime as ort
    import torch
    from transformers.models.d_fine.modeling_d_fine import weighting_function

    staging, cache = Path(staging), Path(cache)
    base, state, aliases = load_source_model(cache)
    wrapper = export_wrapper(base)
    with torch.inference_mode():
        observed_shapes = {name: list(value.shape) for name, value in
                           zip(OUTPUT_SHAPES, wrapper(torch.from_numpy(fixture_input())))}
    if observed_shapes != OUTPUT_SHAPES:
        raise ValueError(f"Unexpected source output shapes: {observed_shapes}")
    combined_path = staging / "detector-with-features.onnx"
    with torch.inference_mode():
        torch.onnx.export(wrapper, torch.from_numpy(fixture_input()), str(combined_path),
                          input_names=["images"], output_names=list(OUTPUT_SHAPES),
                          opset_version=17, dynamo=False, do_constant_folding=False,
                          keep_initializers_as_inputs=False, export_params=True, external_data=False)
    combined = onnx.load(str(combined_path), load_external_data=False)
    onnx.checker.check_model(combined, full_check=True)
    combined = onnx.shape_inference.infer_shapes(combined, check_type=True, strict_mode=True, data_prop=True)
    bind_output_shapes(combined, observed_shapes)
    constant_changes = canonicalize_fp32_constants(combined)
    shape = lambda value: [d.dim_value if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]
    if len(combined.graph.input) != 1 or shape(combined.graph.input[0]) != [1, 3, SIZE, SIZE]:
        raise ValueError("Unexpected static input shape")
    actual_shapes = {v.name: shape(v) for v in combined.graph.output}
    if actual_shapes != OUTPUT_SHAPES:
        raise ValueError(f"Unexpected static output shapes: {actual_shapes}")
    tensor_inventory(combined)
    # Extract dependency closures, so requesting prefix outputs cannot still
    # execute the final trainable heads or downstream postprocessing.
    extractor = onnx.utils.Extractor(combined)
    inference = extractor.extract_model(["images"], ["logits", "boxes"])
    prefix = extractor.extract_model(["images"], list(OUTPUT_SHAPES)[2:])
    graph_reports, artifacts = {}, {}
    for name, model in (("detector.onnx", inference), ("frozen-prefix.onnx", prefix)):
        onnx.checker.check_model(model, full_check=True)
        graph_reports[name] = tensor_inventory(model)
        onnx.save(model, str(staging / name))
        artifacts[name] = {"file": name, "sha256": sha256_file(staging / name),
                           "bytes": (staging / name).stat().st_size,
                           "under25MiB": (staging / name).stat().st_size <= 25 * 1024 * 1024}
    combined_path.unlink()
    mapping = map_parameters(inference, state)
    trainable_names = [f"model.decoder.bbox_embed.2.layers.{index}.{kind}"
                       for index in range(3) for kind in ("weight", "bias")]
    trainable_names += [f"model.decoder.class_embed.2.{kind}" for kind in ("weight", "bias")]
    by_source = {entry["sourceKey"]: entry for entry in mapping}
    if any(name not in by_source for name in trainable_names):
        raise ValueError("Export did not preserve every final-head initializer identity")
    if any(entry["sourceKey"] in trainable_names for entry in map_parameters(prefix, state)):
        raise ValueError("Frozen prefix still contains trainable final-head weights")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(str(staging / name), sess_options=options,
                                    providers=["CPUExecutionProvider"])
                for name in ("detector.onnx", "frozen-prefix.onnx")]
    fixtures = []
    for phase in FIXTURE_PHASES:
        images = fixture_input(phase)
        with torch.inference_mode():
            native = {name: value.numpy().copy() for name, value in zip(OUTPUT_SHAPES, wrapper(torch.from_numpy(images)))}
        actual = {}
        for session in sessions:
            actual.update(zip([item.name for item in session.get_outputs()], session.run(None, {"images": images})))
        comparisons, files = {}, {}
        fixture_dir = staging / "fixtures" / f"pattern-{phase}"
        fixture_dir.mkdir(parents=True)
        for name, values in native.items():
            if list(values.shape) != OUTPUT_SHAPES[name] or values.dtype != np.float32:
                raise ValueError(f"Unexpected native tensor contract: {name}")
            try:
                comparisons[name] = comparison(actual[name], values, name)
            except NumericParityError as error:
                # Preserve all 18 measured comparisons. Shape/nonfinite/source
                # failures still abort without a usable diagnostic manifest.
                comparisons[name] = error.result
            path = fixture_dir / f"{name}.f32"
            path.write_bytes(values.astype("<f4", copy=False).tobytes())
            files[name] = {"file": str(path.relative_to(staging)), "shape": list(values.shape),
                           "dtype": "float32-le", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        fixtures.append({"id": f"dfine-rgb-pattern-v1-phase-{phase}", "phase": phase,
                         "inputSha256": tensor_sha256(images), "outputs": files, "comparisons": comparisons})
    source = {"modelId": MODEL_ID, "revision": SOURCE_REVISION, "file": "model.safetensors",
              "url": SOURCE_URL, "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES,
              "license": "Apache-2.0", "trainingDataset": "COCO-only",
              "attentionConversionFix": "066438d merges c95fb3e from HF pull request #1; Transformers #40257 corrects Nano K slice to [128:256]."}
    projection = weighting_function(base.config.max_num_bins, base.model.decoder.up, base.model.decoder.reg_scale)
    head = {
        "scope": "Final pretrained bbox MLP and class projection only; all upstream tensors and LQE frozen.",
        "parameters": [by_source[name] for name in trainable_names],
        "frozenLqeParameters": [entry for entry in mapping if entry["sourceKey"].startswith("model.decoder.lqe_layers.2.")],
        "projection": projection.detach().numpy().tolist(), "regScale": float(base.model.decoder.reg_scale.item()),
        "topK": 4, "bins": 33, "queries": 300, "hiddenSize": 128, "classes": 80,
        "features": {name: shape_value for name, shape_value in OUTPUT_SHAPES.items() if name not in ("logits", "boxes")},
    }
    head_config = {key: head[key] for key in ("queries", "hiddenSize", "classes", "bins", "regScale", "topK", "projection")}
    head_names = trainable_names + [f"model.decoder.lqe_layers.2.reg_conf.layers.{index}.{kind}"
                                    for index in range(2) for kind in ("weight", "bias")]
    head_weights = {"schemaVersion": 1, "config": head_config,
                    "weights": {name: {"shape": list(state[name].shape),
                                       "data": tensor_array(state[name]).ravel().tolist()} for name in head_names}}
    write_json(staging / "head-weights.json", head_weights)
    artifacts["headWeights"] = {"file": "head-weights.json", "sha256": sha256_file(staging / "head-weights.json"),
                                "bytes": (staging / "head-weights.json").stat().st_size,
                                "under25MiB": (staging / "head-weights.json").stat().st_size <= 25 * 1024 * 1024}
    export_recipe = {"schemaVersion": 1, "id": "dfine-nano-coco-final-head-v1",
                     "input": {"name": "images", "shape": [1, 3, SIZE, SIZE]},
                     "outputs": [{"name": name, "shape": OUTPUT_SHAPES[name]} for name in ("logits", "boxes")],
                     "parameters": [{"parameter": name, "initializers": [by_source[name]["initializerName"]],
                                     "shape": by_source[name]["shape"]} for name in trainable_names]}
    if len({entry["initializers"][0] for entry in export_recipe["parameters"]}) != 8:
        raise ValueError("Independent final-head parameters share an ONNX initializer")
    source_tensors = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": tensor_sha256(value)} for name, value in sorted(state.items())]
    source_documents = {name: {"url": spec[0], "sha256": spec[1], "bytes": spec[2]}
                        for name, spec in DOCUMENTS.items()}
    tool_versions = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
    passed = all(row["passed"] for fixture in fixtures for row in fixture["comparisons"].values())
    status = "transformers-to-onnx-verified" if passed else FAILED_STATUS
    report = {
        "schemaVersion": 1, "passed": passed, "status": status,
        "scope": "Python CPU eval parity and complete static inference export on three procedural fixtures; no model quality, browser execution, native .pth equality, or training claim.",
        "source": source, "code": {"transformersVersion": "4.56.0", "transformersCommit": TRANSFORMERS_COMMIT,
                                   "nativeDfineReferenceCommit": DFINE_COMMIT, "strictLoad": True,
                                   "evalMode": True, "constantFolding": False, "sharedHeadAliases": aliases,
                                   "fp32ScalarConstantEvaluation": constant_changes,
                                   "staticOutputMetadata": "Bound unresolved dimensions to observed source shapes; all ORT fixture output shapes checked."},
        "artifacts": artifacts, "tolerances": TOLERANCES, "fixtures": fixtures,
        "sourceTensorCount": len(state), "sourceTensors": source_tensors,
        "sourceParameterCount": sum(value.numel() for value in base.parameters()),
        "parameterMapping": mapping, "head": head, "exportRecipe": export_recipe, "sourceDocuments": source_documents,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "toolVersions": tool_versions, "executionProvider": "CPUExecutionProvider", "threads": 1},
    }
    manifest = {
        "schemaVersion": 1, "id": "dfine-nano", "task": "object-detection",
        "status": "developer-investigation" if passed else FAILED_STATUS, "passed": passed,
        "source": source, "artifacts": artifacts,
        "input": {"name": "images", "shape": [1, 3, SIZE, SIZE], "dtype": "float32", "color": "RGB",
                  "resize": "bilinear full-image stretch to 640x640; no padding", "rescale": "RGB / 255",
                  "normalize": False, "fixtureAlreadySized": True},
        "outputs": {"logits": {"shape": [1, 300, 80], "meaning": "raw per-query independent-class logits, including frozen LQE correction"},
                    "boxes": {"shape": [1, 300, 4], "format": "cxcywh", "coordinates": "relative to image width/height; not clamped"}},
        "postprocessing": "Outside graph: sigmoid scores; select top query/class pairs; convert cxcywh to xyxy and scale by original size. No NMS or threshold in graph.",
        "labels": [base.config.id2label[index] for index in range(80)],
        "head": head, "exportRecipe": export_recipe, "tolerances": TOLERANCES,
        "fixtureRecipe": {"size": [640, 640], "phases": list(FIXTURE_PHASES),
                          "r": "(7*x+3*y+phase)%256", "g": "(2*x+11*y+3*phase)%256",
                          "b": "(13*x+5*y+7*phase)%256", "layout": "NCHW", "rescale": "float32 RGB / float32(255)"},
        "fixtures": fixtures, "licenses": ["LICENSE.txt", "TRANSFORMERS_LICENSE.txt", "NOTICE.txt", "MODEL_CARD.md"],
    }
    notice = f"""Open Vision Builder — D-FINE Nano attribution and modification record

Public weights: {MODEL_ID}, revision {SOURCE_REVISION}
Source model.safetensors SHA-256: {SOURCE_SHA256}
Source model card: Apache-2.0; dataset: COCO. No Objects365 checkpoint used.
D-FINE source/reference commit: {DFINE_COMMIT}
Transformers implementation: 4.56.0, commit {TRANSFORMERS_COMMIT}
Transformers modeling_d_fine.py copyright: 2025 Baidu Inc and The HuggingFace Inc. team.

The pinned D-FINE README and Apache LICENSE are preserved. The maintainer's
COCO-only checkpoint license clarification is at
https://github.com/Peterande/D-FINE/issues/357#issuecomment-5344034723
The model card independently declares Apache-2.0 for this exact HF model.
The converter source is preserved for lineage only, never executed here.
No native .pth checkpoint was downloaded or loaded, so numerical equivalence
between the HF and original native checkpoint is not asserted.

Changes made downstream:
- Loaded only checksum-verified safetensors; populated 24 explicit shared-head
  aliases and then strictly loaded the complete unmodified PyTorch model.
- Exported the FP32 eval model at fixed [1,3,640,640], ONNX opset 17, with all
  weights embedded. Constant folding and deploy reparameterization disabled.
- Bound unresolved output metadata to measured source output dimensions. Evaluated
  FP64 scalar constants only when every consumer was an existing FP32 Cast;
  those casts are retained and receive the exact same float32 scalar bits.
- Exposed raw logits/normalized cxcywh boxes. No NMS/threshold in the graph.
- Extracted a separate frozen prefix producing final-head inputs and residuals.
  This is for a bounded head-training experiment, not the native training recipe.
- The fixture images are original procedural RGB patterns; no user data or
  third-party photograph is included. They do not establish detector accuracy.

Preserve the model card, both Apache licenses, source metadata, and this
downstream modification notice with derivative model exports. Record any
subsequent changed weights and training recipe separately. This NOTICE was
written downstream; it is not an upstream NOTICE or a new license grant.
"""
    (staging / "NOTICE.txt").write_text(notice, encoding="utf-8")
    for name in DOCUMENTS:
        shutil.copyfile(cache / name, staging / name)
    write_json(staging / "preparation.json", report)
    write_json(staging / "inventory.json", graph_reports)
    write_json(staging / "model.json", manifest)
    return report


def _validate_owned_destination(destination):
    if not destination.exists():
        return
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"Refusing unowned output path: {destination}")
    if not any(destination.iterdir()):
        return
    owned = False
    for filename in ("preparation-state.json", "model.json", "failure.json"):
        marker = destination / filename
        if marker.is_file() and not marker.is_symlink():
            try:
                metadata = json.loads(marker.read_text(encoding="utf-8"))
                owned = owned or metadata.get("source", {}).get("modelId") == MODEL_ID
            except (ValueError, OSError, AttributeError):
                pass
    allowed = set(DOCUMENTS) | {"model.json", "preparation.json", "inventory.json", "head-weights.json",
                                "detector.onnx", "frozen-prefix.onnx", "fixtures", "NOTICE.txt",
                                "failure.json", "preparation-state.json"}
    if not owned or any(path.name not in allowed for path in destination.iterdir()):
        raise ValueError(f"Refusing to clear unowned output directory: {destination}")


def _reset_destination(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    atomic_write_json(destination / "preparation-state.json",
                      {"source": {"modelId": MODEL_ID}, "passed": False, "status": "preparing"})


def _qualification_from_comparisons(report):
    fixtures = report.get("fixtures", [])
    if len(fixtures) != 3 or {fixture.get("phase") for fixture in fixtures} != set(FIXTURE_PHASES):
        raise ValueError("Incomplete fixture comparisons; no diagnostic manifest can be published")
    results = []
    for fixture in fixtures:
        comparisons = fixture.get("comparisons", {})
        if set(comparisons) != set(OUTPUT_SHAPES):
            raise ValueError("Incomplete tensor comparisons; no diagnostic manifest can be published")
        for row in comparisons.values():
            if type(row.get("passed")) is not bool:
                raise ValueError("A comparison has no completed boolean result")
            results.append(row["passed"])
    return all(results)


def _publish_staging(staging, destination):
    # The manifest is last; every payload has already passed source, structural,
    # dtype, finite-value and size checks, even for unqualified numeric evidence.
    for path in staging.iterdir():
        if path.name == "model.json":
            continue
        shutil.move(str(path), str(destination / path.name))
    (staging / "model.json").replace(destination / "model.json")


def prepare(output, cache, diagnostic=DEFAULT_DIAGNOSTIC_DIR):
    """Publish qualified assets only when all original numeric gates pass.

    A completed numeric failure publishes reproducible evidence to the separate
    diagnostic directory and still raises/returns exit 1. This is not a bypass.
    Source, download, shape, finite-value or structural failure publishes no
    usable model manifest anywhere. Prior owned assets are invalidated first.
    """
    output, cache, diagnostic = (Path(path).resolve() for path in (output, cache, diagnostic))
    destinations = (output, diagnostic)
    for left, right in ((output, diagnostic), (output, cache), (diagnostic, cache)):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Output, diagnostics and source cache must be disjoint directories")
    for destination in destinations:
        _validate_owned_destination(destination)
    for destination in destinations:
        _reset_destination(destination)
    try:
        with tempfile.TemporaryDirectory(prefix=".detector-stage-", dir=diagnostic.parent) as temporary:
            staging = Path(temporary)
            report = _build_assets(staging, cache)
            passed = _qualification_from_comparisons(report)
            status = "transformers-to-onnx-verified" if passed else FAILED_STATUS
            report.update(passed=passed, status=status)
            manifest = json.loads((staging / "model.json").read_text(encoding="utf-8"))
            manifest.update(passed=passed, status="developer-investigation" if passed else FAILED_STATUS)
            if not passed:
                manifest["qualificationWarning"] = "INVESTIGATION ONLY: source-to-prefix numeric qualification failed; see all unchanged tolerances and measured comparisons."
            write_json(staging / "preparation.json", report)
            write_json(staging / "model.json", manifest)
            if any(path.stat().st_size > 25 * 1024 * 1024 for path in staging.rglob("*") if path.is_file()):
                raise ValueError("A static artifact exceeds the 25 MiB limit")
            destination = output if passed else diagnostic
            _publish_staging(staging, destination)
            atomic_write_json(destination / "preparation-state.json",
                              {"source": {"modelId": MODEL_ID}, "passed": passed, "status": status})
            if not passed:
                raise QualificationFailure(diagnostic)
            atomic_write_json(diagnostic / "preparation-state.json",
                              {"source": {"modelId": MODEL_ID}, "passed": False, "status": "no-diagnostic-required"})
            return report
    except Exception as error:
        failure = {"schemaVersion": 1, "source": {"modelId": MODEL_ID, "revision": SOURCE_REVISION,
                                                  "sha256": SOURCE_SHA256},
                   "passed": False, "status": FAILED_STATUS if isinstance(error, QualificationFailure) else "preparation-error",
                   "error": str(error), "tolerances": TOLERANCES}
        if isinstance(error, QualificationFailure):
            failure["diagnosticDirectory"] = str(diagnostic)
            atomic_write_json(output / "failure.json", failure)
            atomic_write_json(output / "preparation-state.json", failure)
        else:
            for destination in destinations:
                (destination / "model.json").unlink(missing_ok=True)
                (destination / "preparation.json").unlink(missing_ok=True)
                atomic_write_json(destination / "failure.json", failure)
                atomic_write_json(destination / "preparation-state.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/p1b/detector-assets/prepared")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/p1b/detector-assets/source")
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR,
                        help="Separate failed-numeric evidence directory; failure still exits 1")
    args = parser.parse_args()
    try:
        result = prepare(args.output_dir, args.cache_dir, args.diagnostic_dir)
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, "output": str(args.output_dir), "artifacts": result["artifacts"],
                      "comparison": {item["id"]: item["comparisons"] for item in result["fixtures"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
