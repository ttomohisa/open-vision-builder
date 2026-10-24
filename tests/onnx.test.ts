import { describe, expect, it } from 'vitest';
import { create, fromBinary, toBinary } from '@bufbuild/protobuf';
import { ModelProtoSchema, TensorProtoSchema, ValueInfoProtoSchema } from 'onnx-buf';
import * as ort from 'onnxruntime-web/wasm';
import { exportClassifier } from '../src/ml/onnx';
import type { ModelManifest } from '../src/ml/types';

function vi(name: string, shape: number[]) {
  return create(ValueInfoProtoSchema, { name, type: { value: { case: 'tensorType', value: { elemType: 1, shape: { dim: shape.map(d => ({ value: { case: 'dimValue' as const, value: BigInt(d) } })) } } } } });
}
function tensor(name: string, dims: number[], values: number[]) {
  return create(TensorProtoSchema, { name, dims: dims.map(BigInt), dataType: 1, floatData: values });
}
function fixture() {
  const model = create(ModelProtoSchema, {
    irVersion: 9n, opsetImport: [{ domain: '', version: 17n }],
    graph: {
      name: 'tiny-pretrained-network', input: [vi('images', [1, 2])], output: [vi('logits', [1, 3]), vi('features', [1, 2])],
      initializer: [tensor('frozen', [2, 2], [2, 0, 0, 3]), tensor('oldW', [2, 3], [1, 1, 1, 1, 1, 1]), tensor('oldB', [3], [1, 2, 3])],
      node: [
        { name: 'backbone', opType: 'MatMul', input: ['images', 'frozen'], output: ['features'] },
        { name: 'classifier', opType: 'Gemm', input: ['features', 'oldW', 'oldB'], output: ['logits'] },
      ],
    },
  });
  const manifest = { schemaVersion: 1, id: 'test', input: { name: 'images', shape: [1, 2] }, features: { name: 'features', size: 2 }, originalOutput: 'logits', head: { nodeName: 'classifier', weightName: 'oldW', biasName: 'oldB' } } as ModelManifest;
  return { model, manifest, bytes: toBinary(ModelProtoSchema, model) };
}

describe('complete ONNX export', () => {
  it('keeps the backbone and runs a changed class head in an independent runtime', async () => {
    const { bytes, manifest } = fixture();
    const output = exportClassifier(bytes, manifest, { weights: new Float32Array([1, 2, 3, 4]), bias: new Float32Array([0.1, -0.2]), featureSize: 2, classes: 2 }, ['cat', 'dog']);
    const decoded = fromBinary(ModelProtoSchema, output);
    expect(decoded.graph!.initializer.map(t => t.name)).toContain('frozen');
    expect(decoded.graph!.initializer.map(t => t.name)).not.toContain('oldW');
    expect(decoded.graph!.input.map(t => t.name)).toEqual(['images']);
    expect(decoded.graph!.output.map(t => t.name)).toEqual(['logits']);
    expect(fromBinary(ModelProtoSchema, bytes).graph!.output.length).toBe(2);
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(output, { executionProviders: ['wasm'] });
    try {
      const result = await session.run({ images: new ort.Tensor('float32', new Float32Array([2, 4]), [1, 2]) });
      expect(result.logits.dims).toEqual([1, 2]);
      expect(Number(result.logits.data[0])).toBeCloseTo(40.1, 4);
      expect(Number(result.logits.data[1])).toBeCloseTo(55.8, 4);
    } finally { await session.release(); }
  });
  it('rejects missing external weights, invalid trained values and duplicate labels', () => {
    const { model, manifest, bytes } = fixture();
    const head = { weights: new Float32Array([1, 2, 3, 4]), bias: new Float32Array(2), featureSize: 2, classes: 2 };
    expect(() => exportClassifier(bytes, manifest, { ...head, weights: new Float32Array([NaN, 0, 0, 0]) }, ['a', 'b'])).toThrow();
    expect(() => exportClassifier(bytes, manifest, head, ['a', 'a'])).toThrow();
    model.graph!.initializer[0].externalData = [{ $typeName: 'onnx.StringStringEntryProto', key: 'location', value: 'missing.data' }];
    expect(() => exportClassifier(toBinary(ModelProtoSchema, model), manifest, head, ['a', 'b'])).toThrow();
  });
});
