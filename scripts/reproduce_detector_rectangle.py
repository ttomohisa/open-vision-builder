#!/usr/bin/env python3
"""Measure the known rectangle failure without changing model qualification.

Developer diagnostic only. Exit 0 means complete measurements were written;
sourceParityPassed records the numerical result, and qualifiesDetector is always
false. Invalid inputs, integrity failures and runtime errors exit nonzero. The
original prepare_detector.py gate remains strict and unchanged.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

import prepare_detector as prepare
import diagnose_detector_decoder as decoder
from diagnose_detector_parity import metrics, session_options

ROOT = Path(__file__).resolve().parents[1]
REPORT_KIND = 'dfine-rectangle-diagnostic-v1'
INPUT_SHA256 = 'dad00b5c87bfc67d5b12f1212f2c88aeab24c6944998b2200f1e28ab9c0b42eb'
GRAPH_SHA256 = {
    'detector.onnx': '4f282002c297af2835e038ea928e40cbcf56380321a914f85fd165e5aded8248',
    'frozen-prefix.onnx': '71f3591bcd5f33b6e23d152816e7eea7f81296771625bc932bcf0ccd6a7bdcf3',
}
BOUNDARY_SHAPES = {f'{module}.{side}': [1, 2000, 4 if module == 'encoderBox' and side == 'output' else 128]
                   for module in ('encoderProjection', 'encoderNorm', 'encoderBox') for side in ('input', 'output')}


def rectangle_input():
    image = np.empty((1, 3, 640, 640), dtype=np.float32)
    image[:] = np.array([31, 47, 63], dtype=np.float32)[None, :, None, None]
    for x0, x1, y0, y1, rgb in ((80, 240, 100, 360, (220, 40, 50)),
                               (360, 580, 300, 590, (30, 200, 90)),
                               (16, 48, 560, 624, (60, 80, 230))):
        image[:, :, y0:y1, x0:x1] = np.array(rgb, dtype=np.float32)[None, :, None, None]
    image /= np.float32(255)
    if prepare.tensor_sha256(image) != INPUT_SHA256:
        raise ValueError('Rectangle bytes differ from the recorded fixture')
    return image


def compare_tensor(actual, expected, name):
    shape = prepare.OUTPUT_SHAPES.get(name, BOUNDARY_SHAPES.get(name, decoder.SHAPES.get(name)))
    if shape is None:
        raise ValueError(f'Unknown comparison: {name}')
    for value in (actual, expected):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or list(value.shape) != shape:
            raise ValueError(f'Wrong FP32 shape for {name}')
    tolerance = prepare.TOLERANCES.get(name, decoder.TOLERANCES.get(name, {'atol': 2e-4, 'rtol': 1e-4}))
    return {**metrics(actual, expected, **tolerance), **tolerance, 'shape': shape,
            'nativeSha256': prepare.tensor_sha256(expected), 'ortSha256': prepare.tensor_sha256(actual)}


def diagnose(assets, cache):
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper

    manifest_path = assets / 'model.json'
    manifest = json.loads(manifest_path.read_text())
    source = manifest['source']
    if (source['modelId'], source['revision'], source['sha256']) != (
            prepare.MODEL_ID, prepare.SOURCE_REVISION, prepare.SOURCE_SHA256):
        raise ValueError('Wrong source identity')
    if manifest['tolerances'] != prepare.TOLERANCES:
        raise ValueError('The original source tolerances changed')
    if type(manifest['passed']) is not bool or manifest['status'] != (
            'developer-investigation' if manifest['passed'] else prepare.FAILED_STATUS):
        raise ValueError('Invalid original qualification status')
    for name, digest in GRAPH_SHA256.items():
        path, artifact = assets / name, manifest['artifacts'][name]
        if (artifact['file'] != name or artifact['sha256'] != digest
                or path.stat().st_size != artifact['bytes'] or prepare.sha256_file(path) != digest):
            raise ValueError(f'Pinned graph identity differs: {name}')
    base, state, _ = prepare.load_source_model(cache)
    wrapper, captures, handles = prepare.export_wrapper(base), {}, []
    modules = {'encoderProjection': base.model.enc_output[0], 'encoderNorm': base.model.enc_output[1],
               'encoderBox': base.model.enc_bbox_head}
    decoder_layers = {f'layer{i}': layer for i, layer in enumerate(base.model.decoder.layers)}
    decoder_mlps = {'preBox': base.model.decoder.pre_bbox_head,
                    'corner0': base.bbox_embed[0], 'corner1': base.bbox_embed[1]}

    def hook(name):
        def record(_module, args, result):
            for side, value in (('input', args[0]), ('output', result)):
                captures[f'{name}.{side}'] = value.detach().cpu().numpy().copy()
        return record

    images = rectangle_input()
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        with decoder.capture_modules(decoder_mlps, decoder_layers, captures), torch.inference_mode():
            native = {name: value.cpu().numpy().copy() for name, value in
                      zip(prepare.OUTPUT_SHAPES, wrapper(torch.from_numpy(images)), strict=True)}
    finally:
        for handle in handles:
            handle.remove()
    actual, models = {}, {}
    for name in GRAPH_SHA256:
        model = onnx.load(str(assets / name), load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
        prepare.tensor_inventory(model)
        prepare.map_parameters(model, state)
        session = ort.InferenceSession(model.SerializeToString(),
            sess_options=session_options(ort, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
            providers=['CPUExecutionProvider'])
        output_names = [value.name for value in session.get_outputs()]
        expected_names = ['logits', 'boxes'] if name == 'detector.onnx' else list(prepare.OUTPUT_SHAPES)[2:]
        if output_names != expected_names:
            raise ValueError(f'Unexpected graph outputs: {name}')
        actual.update(zip(output_names, session.run(None, {'images': images}), strict=True))
        models[name] = model
    comparisons = {name: compare_tensor(actual[name], native[name], name) for name in prepare.OUTPUT_SHAPES}

    instrumented = copy.deepcopy(models['frozen-prefix.onnx'])
    nodes = {node.name: node for node in instrumented.graph.node}
    if len(nodes) != len(instrumented.graph.node):
        raise ValueError('Duplicate graph node names')

    def node(name, operator):
        value = nodes.get('/base/model/' + name)
        if value is None or value.op_type != operator:
            raise ValueError(f'Expected {operator} boundary: {name}')
        return value

    projection = node('enc_output/enc_output.0/MatMul', 'MatMul')
    normalization = node('enc_output/enc_output.1/LayerNormalization', 'LayerNormalization')
    mlp_first = node('enc_bbox_head/layers.0/MatMul', 'MatMul')
    mlp_last = node('enc_bbox_head/layers.2/Add', 'Add')
    taps = {'encoderProjection.input': projection.input[0],
            'encoderProjection.output': normalization.input[0],
            'encoderNorm.input': normalization.input[0], 'encoderNorm.output': normalization.output[0],
            'encoderBox.input': mlp_first.input[0], 'encoderBox.output': mlp_last.output[0]}
    decoder_taps = decoder.graph_taps(node)
    taps.update(decoder_taps)
    shapes = {**BOUNDARY_SHAPES, **decoder.SHAPES, **decoder.INTEGER_SHAPES}
    exposed = {value.name for value in instrumented.graph.output}
    for key, name in taps.items():
        if name not in exposed:
            dtype = TensorProto.INT64 if key in decoder.INTEGER_SHAPES else TensorProto.FLOAT
            instrumented.graph.output.append(helper.make_tensor_value_info(name, dtype, shapes[key]))
            exposed.add(name)
    onnx.checker.check_model(instrumented, full_check=True)
    session = ort.InferenceSession(instrumented.SerializeToString(),
        sess_options=session_options(ort, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        providers=['CPUExecutionProvider'])
    observed = dict(zip([value.name for value in session.get_outputs()], session.run(None, {'images': images}), strict=True))
    unchanged = {name: observed[name].dtype == actual[name].dtype and observed[name].shape == actual[name].shape
                 and observed[name].tobytes() == actual[name].tobytes() for name in list(prepare.OUTPUT_SHAPES)[2:]}
    if not all(unchanged.values()):
        raise ValueError('Exposing boundaries changed original prefix outputs')
    traces = {key: compare_tensor(observed[taps[key]], captures[key], key) for key in BOUNDARY_SHAPES}
    identical = {}
    with torch.inference_mode():
        for key, module in modules.items():
            given = observed[taps[f'{key}.input']]
            result = module(torch.from_numpy(given.copy())).cpu().numpy()
            identical[key] = compare_tensor(observed[taps[f'{key}.output']], result, f'{key}.output')
    return {
        'sourceParityPassed': all(value['passedAtOriginalBound'] for value in comparisons.values()),
        'source': source, 'originalQualification': {'passed': manifest['passed'], 'status': manifest['status']},
        'fixture': {'id': 'rectangles', 'shape': list(images.shape), 'dtype': 'float32', 'sha256': INPUT_SHA256},
        'scriptSha256': prepare.sha256_file(Path(__file__)), 'manifestSha256': prepare.sha256_file(manifest_path),
        'graphSha256': GRAPH_SHA256, 'comparisons': comparisons, 'encoderBoundaries': traces,
        'identicalOrtInput': identical, 'instrumentationBitIdentical': unchanged,
        'decoder': decoder.report(decoder_mlps, decoder_layers, captures, observed, decoder_taps, compare_tensor),
        'decoderScriptSha256': prepare.sha256_file(Path(decoder.__file__)),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in prepare.REQUIRED_VERSIONS},
            'provider': 'CPUExecutionProvider', 'threads': 1, 'ortOptimization': 'ORT_ENABLE_ALL'},
        'scope': 'One known synthetic rectangle input; CPU forward and local encoder/decoder diagnosis only. No browser training, quality, new heldout evaluation or detector qualification.',
    }


def atomic_report(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def execute(output, assets, cache):
    output, assets, cache = Path(output), Path(assets), Path(cache)
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Refusing a symlink report destination')
    if output.resolve().is_relative_to(assets.resolve()) or output.resolve().is_relative_to(cache.resolve()):
        raise ValueError('Report must not overwrite input assets')
    if output.exists():
        try:
            old = json.loads(output.read_text())
            owned = isinstance(old, dict) and old.get('kind') == REPORT_KIND
        except (ValueError, OSError):
            owned = False
        if not owned:
            raise ValueError('Refusing to overwrite an unrelated file')
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {'schemaVersion': 1, 'kind': REPORT_KIND, 'qualifiesDetector': False}
    atomic_report(output, {**header, 'status': 'running'})
    try:
        report = diagnose(assets, cache)
        atomic_report(output, {**report, **header, 'status': 'completed'})
    except Exception as error:
        atomic_report(output, {**header, 'status': 'error', 'error': str(error)})
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({'report': str(output), 'status': 'completed', 'qualifiesDetector': False,
                      'sourceParityPassed': report['sourceParityPassed']}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, default=prepare.DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'artifacts/p1b/detector-assets/source')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/p1c/rectangle-reproduction.json')
    args = parser.parse_args()
    return execute(args.output, args.assets_dir, args.cache_dir)


if __name__ == '__main__':
    raise SystemExit(main())
