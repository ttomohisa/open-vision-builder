import { describe, expect, it } from 'vitest';
import { create, fromBinary, toBinary } from '@bufbuild/protobuf';
import { ModelProtoSchema, TensorProtoSchema, ValueInfoProtoSchema } from 'onnx-buf';
import * as ort from 'onnxruntime-web/wasm';
import { exportDFineOnnx, type DFineExportRecipe } from '../src/ml/detection/export';

function info(name: string, shape: number[]) {
  return create(ValueInfoProtoSchema, { name, type: { value: { case: 'tensorType', value: { elemType: 1, shape: { dim: shape.map(n => ({ value: { case: 'dimValue' as const, value: BigInt(n) } })) } } } } });
}
function tensor(name: string, shape: number[], values: number[]) {
  return create(TensorProtoSchema, { name, dataType: 1, dims: shape.map(BigInt), floatData: values });
}
function fixture() {
  const model = create(ModelProtoSchema, {
    irVersion: 8n, opsetImport: [{ domain: '', version: 17n }],
    graph: {
      name: 'complete-detection-parameter-substitution-regression',
      input: [info('images', [1, 2])], output: [info('logits', [1, 2]), info('boxes', [1, 4]), info('features', [1, 2])],
      initializer: [
        tensor('frozen.backbone', [2, 2], [2, 0, 0, 3]),
        tensor('class.weight', [2, 2], [1, 0, 0, 1]), tensor('class.bias', [2], [0, 0]),
        tensor('bbox.weight', [2, 4], [1, 0, 0, 0, 0, 1, 0, 0]), tensor('bbox.bias', [4], [0, 0, 1, 1]),
      ],
      node: [
        { name: 'backbone', opType: 'MatMul', input: ['images', 'frozen.backbone'], output: ['features'] },
        { name: 'classifier', opType: 'Gemm', input: ['features', 'class.weight', 'class.bias'], output: ['logits'] },
        { name: 'localization', opType: 'Gemm', input: ['features', 'bbox.weight', 'bbox.bias'], output: ['boxes'] },
      ],
    },
  });
  const recipe: DFineExportRecipe = {
    schemaVersion: 1, id: 'dfine-nano-coco-final-head-v1',
    input: { name: 'images', shape: [1, 2] }, outputs: [{ name: 'logits', shape: [1, 2] }, { name: 'boxes', shape: [1, 4] }],
    parameters: [
      { parameter: 'class.weight', initializers: ['class.weight'], shape: [2, 2] },
      { parameter: 'class.bias', initializers: ['class.bias'], shape: [2] },
      { parameter: 'bbox.weight', initializers: ['bbox.weight'], shape: [2, 4] },
      { parameter: 'bbox.bias', initializers: ['bbox.bias'], shape: [4] },
    ],
  };
  const updates = {
    'class.weight': { shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) },
    'class.bias': { shape: [2], data: new Float32Array([0.1, -0.2]) },
    'bbox.weight': { shape: [2, 4], data: new Float32Array([0.1, 0, 0.2, 0, 0, 0.1, 0, 0.2]) },
    'bbox.bias': { shape: [4], data: new Float32Array([0.1, 0.2, 0.3, 0.4]) },
  };
  return { model, recipe, updates, bytes: toBinary(ModelProtoSchema, model) };
}

describe('complete detector ONNX export', () => {
  it('runs updated classification and localization with frozen weights in a fresh runtime', async () => {
    const { bytes, recipe, updates } = fixture();
    const exported = exportDFineOnnx(bytes, recipe, updates);
    const parsed = fromBinary(ModelProtoSchema, exported);
    expect(parsed.graph!.input.map(x => x.name)).toEqual(['images']);
    expect(parsed.graph!.output.map(x => x.name)).toEqual(['logits', 'boxes']);
    const before = fromBinary(ModelProtoSchema, bytes);
    expect(parsed.graph!.initializer.find(x => x.name === 'frozen.backbone')).toEqual(before.graph!.initializer[0]);
    expect(parsed.graph!.node).toEqual(before.graph!.node);
    expect(before.graph!.output).toHaveLength(3);
    expect(before.graph!.initializer.find(x => x.name === 'class.weight')!.floatData).toEqual([1, 0, 0, 1]);
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(exported, { executionProviders: ['wasm'] });
    try {
      const output = await session.run({ images: new ort.Tensor('float32', new Float32Array([2, 4]), [1, 2]) });
      expect(output.logits.type).toBe('float32');
      expect(output.boxes.type).toBe('float32');
      expect(Array.from(output.logits.data as Float32Array)).toEqual([expect.closeTo(40.1, 4), expect.closeTo(55.8, 4)]);
      expect(Array.from(output.boxes.data as Float32Array)).toEqual([expect.closeTo(0.5, 5), expect.closeTo(1.4, 5), expect.closeTo(1.1, 5), expect.closeTo(2.8, 5)]);
    } finally { await session.release(); }
  });

  it('rejects missing, extra, nonfinite, or incorrectly shaped trained tensors', () => {
    const { bytes, recipe, updates } = fixture();
    const missing = { ...updates } as Record<string, typeof updates['class.bias']>;
    delete missing['class.bias'];
    expect(() => exportDFineOnnx(bytes, recipe, missing)).toThrow(/parameter/i);
    expect(() => exportDFineOnnx(bytes, recipe, { ...updates, unexpected: updates['class.bias'] })).toThrow(/parameter/i);
    expect(() => exportDFineOnnx(bytes, recipe, { ...updates, 'class.bias': { shape: [2], data: new Float32Array([NaN, 0]) } })).toThrow(/finite/i);
    expect(() => exportDFineOnnx(bytes, recipe, { ...updates, 'class.weight': { shape: [4], data: new Float32Array(4) } })).toThrow(/shape/i);
  });

  it('rejects incomplete models and conflicting mappings instead of exporting a partial artifact', () => {
    const { bytes, model, recipe, updates } = fixture();
    model.graph!.initializer[0].externalData = [{ $typeName: 'onnx.StringStringEntryProto', key: 'location', value: 'unavailable.bin' }];
    expect(() => exportDFineOnnx(toBinary(ModelProtoSchema, model), recipe, updates)).toThrow(/external/i);
    expect(() => exportDFineOnnx(bytes, { ...recipe, parameters: recipe.parameters.map((p, i) => i === 0 ? { ...p, initializers: ['missing.weight'] } : p) }, updates)).toThrow(/initializer/i);
    expect(() => exportDFineOnnx(bytes, { ...recipe, parameters: recipe.parameters.map((p, i) => i === 1 ? { ...p, initializers: ['class.weight'] } : p) }, updates)).toThrow(/mapping|initializer/i);
    expect(() => exportDFineOnnx(bytes, { ...recipe, input: { name: 'not-images', shape: [1, 2] } }, updates)).toThrow(/input/i);
    expect(() => exportDFineOnnx(bytes, { ...recipe, outputs: [{ name: 'logits', shape: [1, 80] }, recipe.outputs[1]] }, updates)).toThrow(/output/i);
  });

  it('writes every explicitly mapped shared alias without mutating other frozen weights', () => {
    const { model, recipe, updates } = fixture();
    model.graph!.initializer.push(tensor('class.bias.alias', [2], [0, 0]));
    recipe.parameters[1].initializers.push('class.bias.alias');
    const output = fromBinary(ModelProtoSchema, exportDFineOnnx(toBinary(ModelProtoSchema, model), recipe, updates));
    const get = (name: string) => output.graph!.initializer.find(x => x.name === name)!;
    expect(get('class.bias.alias').rawData).toEqual(get('class.bias').rawData);
    expect(get('class.bias').floatData).toEqual([]);
    expect(get('frozen.backbone').floatData).toEqual([2, 0, 0, 3]);
  });
});
