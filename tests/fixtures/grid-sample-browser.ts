// Test-only browser entry. Vite bundles the same installed ONNXModel used by
// production; no operator reimplementation or injected numerical result.
import { create, toBinary } from '@bufbuild/protobuf';
import { defaultDevice, grad, numpy as np, tree } from '@jax-js/jax';
import { ONNXModel } from '@jax-js/onnx';
import { AttributeProto_AttributeType, ModelProtoSchema } from 'onnx-buf';
import type reference from './grid-sample-pytorch.json';

type Fixture = typeof reference.cases[number];

function valueInfo(name: string, shape: number[]) {
  return {
    name,
    type: { value: { case: 'tensorType' as const, value: {
      elemType: 1,
      shape: { dim: shape.map(dimension => ({ value: { case: 'dimValue' as const, value: BigInt(dimension) } })) },
    } } },
  };
}

export async function runGridSampleCases(fixtures: Fixture[]) {
  defaultDevice('wasm');
  const results = [];
  for (const fixture of fixtures) {
    const bytes = toBinary(ModelProtoSchema, create(ModelProtoSchema, {
      irVersion: 9n, opsetImport: [{ domain: '', version: 17n }],
      graph: {
        name: 'grid-sample-browser-reference',
        input: [valueInfo('features', fixture.featureShape), valueInfo('grid', fixture.gridShape)],
        output: [valueInfo('output', fixture.outputShape)],
        node: [{ opType: 'GridSample', input: ['features', 'grid'], output: ['output'], attribute: [
          { name: 'mode', type: AttributeProto_AttributeType.STRING, s: new TextEncoder().encode('bilinear') },
          { name: 'padding_mode', type: AttributeProto_AttributeType.STRING, s: new TextEncoder().encode('zeros') },
          { name: 'align_corners', type: AttributeProto_AttributeType.INT, i: BigInt(fixture.alignCorners) },
        ] }],
      },
    }));
    const model = new ONNXModel(bytes);
    try {
      const inputs = {
        features: np.array(new Float32Array(fixture.features), { shape: fixture.featureShape, dtype: np.float32 }),
        grid: np.array(new Float32Array(fixture.grid), { shape: fixture.gridShape, dtype: np.float32 }),
      };
      const output = model.run(tree.ref(inputs)).output;
      const outputShape = output.shape;
      const actualOutput = await output.data();
      const gradients = grad((operands: typeof inputs) => model.run(operands).output.mul(
        np.array(new Float32Array(fixture.cotangent), { shape: fixture.outputShape, dtype: np.float32 }),
      ).sum())(inputs);
      const [features, grid] = await Promise.all([gradients.features.data(), gradients.grid.data()]);
      results.push({
        name: fixture.name, outputShape,
        actual: { output: Array.from(actualOutput), featureGradient: Array.from(features), gridGradient: Array.from(grid) },
      });
    } finally { model.dispose(); }
  }
  return { backend: 'wasm', userAgent: navigator.userAgent, crossOriginIsolated, results };
}
