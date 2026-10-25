"""Decoder observations for the pinned rectangle diagnostic, never qualification."""
from contextlib import contextmanager

import numpy as np

import prepare_detector as prepare

LAYER_INPUT_SHAPES = {
    'hidden_states': [1, 300, 128], 'position_embeddings': [1, 300, 128],
    'reference_points': [1, 300, 1, 4], 'encoder_hidden_states': [1, 2000, 128],
}
SHAPES = {f'layer{i}.{key}': shape for i in range(3)
          for key, shape in {**LAYER_INPUT_SHAPES, 'output': [1, 300, 128]}.items()}
SHAPES.update({f'{head}.{side}': [1, 300, 128 if side == 'input' else (4 if head == 'preBox' else 132)]
               for head in ('preBox', 'corner0', 'corner1') for side in ('input', 'output')})
INTEGER_SHAPES = {'queryIndices': [1, 300], **{f'spatial{i}': [] for i in range(4)}}
TOLERANCES = {name: {'atol': 2e-5 if name.endswith('.reference_points') else 2e-4, 'rtol': 1e-4}
              for name in SHAPES}


def _spatial_shapes(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.shape != (2, 2)
            or value.tolist() != [[40, 40], [20, 20]]):
        raise ValueError('Unexpected decoder spatial operands')
    return value


def _fp32(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or list(value.shape) != shape
            or not np.isfinite(value).all()):
        raise ValueError(f'Invalid decoder operand: {name}')
    return value


@contextmanager
def capture_modules(mlps, layers, captured):
    """Own every observed array and detach hooks before any local replay."""
    handles = []

    def mlp_hook(name):
        def record(_module, args, result):
            if len(args) != 1 or f'{name}.input' in captured:
                raise ValueError(f'Unexpected repeated/changed MLP call: {name}')
            captured[f'{name}.input'] = args[0].detach().cpu().numpy().copy()
            captured[f'{name}.output'] = result.detach().cpu().numpy().copy()
        return record

    def layer_hook(name):
        def record(_module, args, kwargs, result):
            if args or f'{name}.output' in captured:
                raise ValueError(f'Unexpected repeated/positional decoder call: {name}')
            if kwargs['encoder_attention_mask'] is not None or kwargs['output_attentions'] is not False:
                raise ValueError('Unexpected decoder mask or attention path')
            spatial = _spatial_shapes(kwargs['spatial_shapes'].detach().cpu().numpy().copy())
            if [tuple(row) for row in spatial.tolist()] != kwargs['spatial_shapes_list']:
                raise ValueError('Native spatial tensor and list disagree')
            captured[f'{name}.spatial_shapes'] = spatial
            for key in LAYER_INPUT_SHAPES:
                captured[f'{name}.{key}'] = kwargs[key].detach().cpu().numpy().copy()
            captured[f'{name}.output'] = result[0].detach().cpu().numpy().copy()
        return record

    try:
        for name, module in mlps.items():
            handles.append(module.register_forward_hook(mlp_hook(name)))
        for name, module in layers.items():
            handles.append(module.register_forward_hook(layer_hook(name), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def graph_taps(node):
    """Resolve complete module operands using checked, pinned graph node names."""
    taps = {}
    memory = node('Concat_2', 'Concat').output[0]
    for i, suffix in enumerate(('', '_9', '_18')):
        root = f'decoder/layers.{i}/'
        taps.update({
            f'layer{i}.hidden_states': node(root + 'self_attn/v_proj/MatMul', 'MatMul').input[0],
            f'layer{i}.position_embeddings': node(root + 'self_attn/Add', 'Add').input[1],
            f'layer{i}.reference_points': node(f'decoder/Unsqueeze{suffix}', 'Unsqueeze').output[0],
            f'layer{i}.encoder_hidden_states': memory,
            f'layer{i}.output': node(root + 'final_layer_norm/LayerNormalization', 'LayerNormalization').output[0],
        })
    for name, path in (('preBox', 'pre_bbox_head'), ('corner0', 'bbox_embed.0'), ('corner1', 'bbox_embed.1')):
        root = f'decoder/{path}/'
        taps[f'{name}.input'] = node(root + 'layers.0/MatMul', 'MatMul').input[0]
        taps[f'{name}.output'] = node(root + 'layers.2/Add', 'Add').output[0]
    taps['queryIndices'] = node('TopK', 'TopK').output[1]
    for i, suffix in enumerate(('', '_1', '_2', '_3')):
        taps[f'spatial{i}'] = node(f'Gather{suffix}', 'Gather').output[0]
    return taps


def replay_layer(layer, operands, spatial_shapes):
    """Pass every live ORT operand to the unchanged original PyTorch layer."""
    import torch

    spatial_shapes = _spatial_shapes(spatial_shapes)
    if set(operands) != set(LAYER_INPUT_SHAPES):
        raise ValueError('Incomplete decoder replay operands')
    inputs = {key: torch.from_numpy(_fp32(operands[key], shape, key).copy())
              for key, shape in LAYER_INPUT_SHAPES.items()}
    with torch.inference_mode():
        result = layer(**inputs, spatial_shapes=torch.from_numpy(spatial_shapes.copy()),
            spatial_shapes_list=[tuple(row) for row in spatial_shapes.tolist()],
            encoder_attention_mask=None, output_attentions=False)[0]
    return result.detach().cpu().numpy().copy()


def consumed_query_order(native_memory, native_consumed, indices):
    """Check actual consumed embedding order; identical rows can hide index ties."""
    _fp32(native_memory, [1, 2000, 128], 'native encoder memory')
    _fp32(native_consumed, [1, 300, 128], 'native selected queries')
    if (not isinstance(indices, np.ndarray) or indices.dtype != np.int64 or indices.shape != (1, 300)
            or np.any(indices < 0) or np.any(indices >= 2000) or np.unique(indices).size != 300):
        raise ValueError('Invalid query indices')
    selected = np.take_along_axis(native_memory, np.broadcast_to(indices[..., None], (1, 300, 128)), axis=1)
    return {'matchesConsumedEmbeddings': selected.tobytes() == native_consumed.tobytes(),
            'indicesSha256': prepare.tensor_sha256(indices), 'uniqueIndices': 300,
            'scope': 'ORT indices gather the actual native consumed embeddings in order. Duplicate identical encoder rows can make native index identity ambiguous. No query realignment.'}


def report(mlps, layers, captured, observed, taps, compare):
    import torch

    boundaries = {name: compare(observed[taps[name]], captured[name], name) for name in SHAPES}
    spatial_values = [observed[taps[f'spatial{i}']] for i in range(4)]
    if any(value.dtype != np.int64 or value.shape != () for value in spatial_values):
        raise ValueError('Invalid ONNX spatial scalar')
    spatial = _spatial_shapes(np.array(spatial_values, dtype=np.int64).reshape(2, 2))
    if any(not np.array_equal(captured[f'{name}.spatial_shapes'], spatial) for name in layers):
        raise ValueError('Native and ONNX spatial operands disagree')
    replays, inputs = {}, {}
    for name, layer in layers.items():
        operands = {key: observed[taps[f'{name}.{key}']] for key in LAYER_INPUT_SHAPES}
        replayed = replay_layer(layer, operands, spatial)
        replays[name] = compare(observed[taps[f'{name}.output']], replayed, f'{name}.output')
        inputs[name] = {key: prepare.tensor_sha256(value) for key, value in operands.items()}
    with torch.inference_mode():
        for name, module in mlps.items():
            given = observed[taps[f'{name}.input']]
            replayed = module(torch.from_numpy(given.copy())).cpu().numpy().copy()
            replays[name] = compare(observed[taps[f'{name}.output']], replayed, f'{name}.output')
            inputs[name] = {'input': prepare.tensor_sha256(given)}
    return {'boundaries': boundaries, 'identicalOrtInput': replays, 'replayInputSha256': inputs,
            'spatialShapes': spatial.tolist(), 'taps': taps,
            'queryOrder': consumed_query_order(captured['encoderNorm.output'], captured['layer0.hidden_states'],
                                                observed[taps['queryIndices']]),
            'referenceMeaning': 'Per-layer reference_points are the refined coordinates actually consumed. The existing references export is the fixed pre-bbox anchor after layer 0.',
            'scope': 'Known rectangle, original FP32 module replay on identical ORT operands. Numerical observations only; no detector qualification or quality claim.'}
