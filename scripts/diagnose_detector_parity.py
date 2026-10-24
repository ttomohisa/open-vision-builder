#!/usr/bin/env python3
"""Explain the failed D-FINE prefix gate; this command never qualifies a model.

Run strict prepare_detector.py first. This reproducer consumes its checksummed
unqualified developer artifacts and the same pinned public safetensors. Default
diagnostics locate the first Conv divergence and separate upstream errors from
bbox-MLP implementation errors. Optional backend/FP64 investigations preserve
the original thresholds and do not modify any product graph or source weights.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def float32_dot_product_bound(products):
    """Standard gamma_n * sum(abs(a_i*b_i)) FP32 dot-product bound.

    Float32 operands multiply exactly in float64; float64 summation supplies a
    much more accurate local reference. The bound is deliberately conservative
    and is diagnostic evidence, not a replacement model-parity tolerance.
    """
    products = np.asarray(products, dtype=np.float64)
    if not products.size or not np.isfinite(products).all() or products.size >= 2**24:
        raise ValueError("The dot product must have a finite nonempty bounded number of terms")
    unit_roundoff = 2**-24
    gamma = products.size * unit_roundoff / (1 - products.size * unit_roundoff)
    return float(gamma * np.abs(products).sum(dtype=np.float64))


def metrics(actual, expected, atol=2e-4, rtol=1e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid diagnostic tensor")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bound = atol + rtol * np.abs(expected.astype(np.float64))
    return {"passedAtOriginalBound": bool(np.all(error <= bound)), "maxAbsoluteError": float(error.max(initial=0)),
            "meanAbsoluteError": float(error.mean()), "maxToleranceRatio": float((error / bound).max(initial=0)),
            "failedElements": int(np.count_nonzero(error > bound)), "differentElements": int(np.count_nonzero(actual != expected)),
            "elements": actual.size, "referenceMinimum": float(expected.min()), "referenceMaximum": float(expected.max())}


def session_options(ort, level):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = level
    return options


def diagnose(assets, cache, include_matrix=False, include_fp64=False, fp64_path=None):
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper
    import prepare_detector as p

    manifest = json.loads((assets / "model.json").read_text(encoding="utf-8"))
    if manifest["source"]["sha256"] != p.SOURCE_SHA256 or manifest["source"]["revision"] != p.SOURCE_REVISION:
        raise ValueError("Diagnostic artifacts are not from the pinned source checkpoint")
    artifact = manifest["artifacts"]["frozen-prefix.onnx"]
    if artifact["file"] != "frozen-prefix.onnx":
        raise ValueError("Unexpected diagnostic graph filename")
    prefix_path = assets / artifact["file"]
    if not prefix_path.is_file():
        raise ValueError("Run strict preparation to generate diagnostic artifacts first")
    p.download_pinned("https://unused.invalid/", prefix_path, artifact["sha256"], artifact["bytes"])
    base, state, _aliases = p.load_source_model(cache)
    wrapper = p.export_wrapper(base)
    captures = {}

    def hook(name):
        def record(_module, arguments, result):
            captures[name + ".input"] = arguments[0].detach().numpy().copy()
            captures[name + ".output"] = result.detach().numpy().copy()
        return record

    stem = base.model.backbone.model.embedder.stem1
    handles = [getattr(stem, name).register_forward_hook(hook(name))
               for name in ("convolution", "normalization", "lab")]
    for index in (0, 1):
        handles.append(base.bbox_embed[index].register_forward_hook(hook(f"head{index}")))
        for layer in range(3):
            handles.append(base.bbox_embed[index].layers[layer].register_forward_hook(hook(f"head{index}.linear{layer}")))
    images = p.fixture_input(0)
    with torch.inference_mode():
        native = {name: value.numpy().copy() for name, value in zip(p.OUTPUT_SHAPES, wrapper(torch.from_numpy(images)))}
    for handle in handles:
        handle.remove()
    model = onnx.load(str(prefix_path), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    p.tensor_inventory(model)
    p.map_parameters(model, state)
    nodes = {node.name: node for node in model.graph.node}
    names = {
        "convolution.output": "/base/model/backbone/model/embedder/stem1/convolution/Conv_output_0",
        "normalization.output": "/base/model/backbone/model/embedder/stem1/normalization/Add_1_output_0",
        "lab.output": "/base/model/backbone/model/embedder/stem1/lab/Add_output_0",
    }
    for index in (0, 1):
        root = f"/base/model/decoder/bbox_embed.{index}"
        names[f"head{index}.input"] = nodes[root + "/layers.0/MatMul"].input[0]
        names[f"head{index}.output"] = nodes[root + "/layers.2/Add"].output[0]
        for layer in range(3):
            names[f"head{index}.linear{layer}.input"] = nodes[root + f"/layers.{layer}/MatMul"].input[0]
            names[f"head{index}.linear{layer}.output"] = nodes[root + f"/layers.{layer}/Add"].output[0]
    exposed = {value.name for value in model.graph.output}
    for key, name in names.items():
        if name not in exposed:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, captures[key].shape))
            exposed.add(name)
    onnx.checker.check_model(model, full_check=True)
    observed = {}
    for level in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
        session = ort.InferenceSession(model.SerializeToString(), sess_options=session_options(ort, level),
                                       providers=["CPUExecutionProvider"])
        observed[str(level)] = dict(zip(names, session.run(list(names.values()), {"images": images})))
    first = observed[str(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]
    reference, actual = captures["convolution.output"], first["convolution.output"]
    index = np.unravel_index(np.argmax(np.abs(actual - reference)), actual.shape)
    batch, channel, y, x = index
    conv = stem.convolution
    padded = np.pad(captures["convolution.input"].astype(np.float64),
                    ((0, 0), (0, 0), (conv.padding[0], conv.padding[0]), (conv.padding[1], conv.padding[1])))
    window = padded[batch, :, y * conv.stride[0]:y * conv.stride[0] + conv.kernel_size[0],
                    x * conv.stride[1]:x * conv.stride[1] + conv.kernel_size[1]]
    products = window * conv.weight.detach().numpy()[channel].astype(np.float64)
    reference64 = products.sum(dtype=np.float64)
    report = {
        "schemaVersion": 1, "qualifiesDetector": False,
        "source": manifest["source"], "prefixArtifact": artifact, "fixturePhase": 0,
        "scope": "Numerical diagnosis only; the unchanged full 3-fixture/6-output qualification is in preparation.json.",
        "firstStageProfile": "CPU FP32, one thread, ORT optimizations disabled; intermediate outputs exposed",
        "firstStages": {name: metrics(first[name + ".output"], captures[name + ".output"])
                        for name in ("convolution", "normalization", "lab")},
        "firstConvolutionWorstDifference": {
            "outputIndex": [int(value) for value in index], "termCount": products.size,
            "torch": float(reference[index]), "onnx": float(actual[index]), "float64DotProduct": float(reference64),
            "torchAbsoluteErrorVsFloat64": float(abs(reference[index] - reference64)),
            "onnxAbsoluteErrorVsFloat64": float(abs(actual[index] - reference64)),
            "gammaNBoundOnFloat32Summation": float32_dot_product_bound(products),
            "weightsExactSource": np.array_equal(conv.weight.detach().numpy(),
                                                  state["model.backbone.model.embedder.stem1.convolution.weight"].numpy()),
        },
    }
    mlp = observed[str(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]
    report["bboxMlpTraces"] = {name: metrics(mlp[name], captures[name]) for name in names if name.startswith("head")}
    report["bboxMlpOnIdenticalOrtInput"] = {}
    for index in (0, 1):
        with torch.inference_mode():
            value = base.bbox_embed[index](torch.from_numpy(mlp[f"head{index}.input"])).numpy()
        report["bboxMlpOnIdenticalOrtInput"][str(index)] = metrics(mlp[f"head{index}.output"], value)
    if include_matrix:
        references = {"mkldnn-enabled": native}
        was_enabled = torch.backends.mkldnn.enabled
        try:
            torch.backends.mkldnn.enabled = False
            with torch.inference_mode():
                references["mkldnn-disabled"] = {name: value.numpy().copy()
                    for name, value in zip(p.OUTPUT_SHAPES, wrapper(torch.from_numpy(images)))}
        finally:
            torch.backends.mkldnn.enabled = was_enabled
        report["backendMatrix"] = {}
        for level in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                      ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            session = ort.InferenceSession(str(prefix_path), sess_options=session_options(ort, level), providers=["CPUExecutionProvider"])
            values = dict(zip(list(p.OUTPUT_SHAPES)[2:], session.run(None, {"images": images})))
            report["backendMatrix"][str(level)] = {backend: {name: metrics(value, ref[name], **p.TOLERANCES[name])
                for name, value in values.items()} for backend, ref in references.items()}
    if include_fp64:
        report["fp64PrefixInvestigation"] = fp64_prefix(base, state, images, native, fp64_path)
    return report


def fp64_prefix(base, state, images, native, path):
    """Reproduce the rejected FP64 diagnostic; never call it a product asset."""
    import onnx
    import onnxruntime as ort
    import torch
    import prepare_detector as p
    import generate_detector_training_reference as training
    from transformers.models.d_fine.modeling_d_fine import weighting_function

    double_wrapper = p.export_wrapper(copy.deepcopy(base).double())
    with torch.inference_mode():
        wide = {name: value.float().numpy().copy() for name, value in
                zip(p.OUTPUT_SHAPES, double_wrapper(torch.from_numpy(images).double()))}
    weights = {name: value for name, value in state.items() if name.startswith(
        ("model.decoder.bbox_embed.2.", "model.decoder.class_embed.2.", "model.decoder.lqe_layers.2."))}
    config = {"queries": 300, "hiddenSize": 128, "classes": 80, "bins": 33, "regScale": 4, "topK": 4,
              "projection": weighting_function(32, base.model.decoder.up, base.model.decoder.reg_scale)}
    with torch.inference_mode():
        mixed = training.forward({name: torch.from_numpy(wide[name][0]) for name in list(p.OUTPUT_SHAPES)[2:]}, weights, config)
    wide.update(logits=mixed["logits"].numpy()[None], boxes=mixed["boxes"].numpy()[None])
    result = {"productCandidate": False, "headDtypes": sorted({str(value.dtype) for value in weights.values()}),
              "sourceComparison": {name: metrics(wide[name], native[name], **p.TOLERANCES[name]) for name in native}}

    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapped = double_wrapper

        def forward(self, images):
            return tuple(value.float() for value in self.wrapped(images.double())[2:])

    try:
        with torch.inference_mode():
            torch.onnx.export(Prefix().eval(), torch.from_numpy(images), str(path), input_names=["images"],
                              output_names=list(p.OUTPUT_SHAPES)[2:], opset_version=17, dynamo=False,
                              do_constant_folding=False, keep_initializers_as_inputs=False, export_params=True, external_data=False)
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
        result.update(exportBytes=path.stat().st_size, under25MiB=path.stat().st_size <= 25 * 1024 * 1024,
                      exportedTrainableHeadNames=[tensor.name for tensor in model.graph.initializer
                          if ".bbox_embed.2." in tensor.name or ".class_embed.2." in tensor.name])
        session = ort.InferenceSession(str(path), sess_options=session_options(ort, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                                       providers=["CPUExecutionProvider"])
        actual = dict(zip(list(p.OUTPUT_SHAPES)[2:], session.run(None, {"images": images})))
        result["onnxVersusFp64NativeThenFp32"] = {name: metrics(value, wide[name], **p.TOLERANCES[name]) for name, value in actual.items()}
        result["exportStatus"] = "executed"
    except Exception as error:
        result.update(exportStatus="blocked", exportError=str(error))
    return result


def main():
    import prepare_detector as p

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=p.DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/p1b/detector-assets/source")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/p1b/detector-assets/parity-diagnosis.json")
    parser.add_argument("--backend-matrix", action="store_true")
    parser.add_argument("--fp64-prefix", action="store_true", help="Reproduce the rejected non-product FP64 diagnostic")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = diagnose(args.assets_dir, args.cache_dir, args.backend_matrix, args.fp64_prefix,
                          args.output.with_suffix(".fp64-prefix.onnx"))
        p.atomic_write_json(args.output, report)
    except Exception as error:
        p.atomic_write_json(args.output, {"qualifiesDetector": False, "status": "diagnostic-error", "error": str(error)})
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"qualifiesDetector": False, "report": str(args.output),
                      "firstConvolution": report["firstConvolutionWorstDifference"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
