// A prerequisite operator probe, not a D-FINE model or training benchmark.
// Exit 1 means forward/backward disagrees with the hand-derived reference.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { create, toBinary } from '@bufbuild/protobuf';
import { defaultDevice, grad, numpy as np, tree } from '@jax-js/jax';
import { ONNXModel } from '@jax-js/onnx';
import { AttributeProto_AttributeType, ModelProtoSchema } from 'onnx-buf';

const require = createRequire(import.meta.url);
function packageVersion(name) {
  let directory = dirname(require.resolve(name));
  while (true) {
    const filename = join(directory, 'package.json');
    if (existsSync(filename)) {
      const manifest = JSON.parse(readFileSync(filename, 'utf8'));
      if (manifest.name === name) return manifest.version;
    }
    const parent = dirname(directory);
    if (parent === directory) throw new Error(`Cannot find the version of ${name}`);
    directory = parent;
  }
}

const versions = Object.fromEntries([
  '@jax-js/jax', '@jax-js/onnx', '@bufbuild/protobuf', 'onnx-buf',
].map((name) => [name, packageVersion(name)]));

// Frozen before execution. All coordinates avoid integer-pixel derivative knots.
const tolerance = { output: 1e-5, featureGradient: 1e-5, gridGradient: 1e-5 };
const featureValues = new Float32Array([
  10, 20, 30, 40,
  50, 60, 70, 80,
  90, 100, 110, 120,
  130, 140, 150, 160,
]);
const fixtures = [
  { name: 'interior', grid: [-3 / 8, -3 / 8], output: 47.5, featureGradient: [[0, 1 / 16], [1, 3 / 16], [4, 3 / 16], [5, 9 / 16]], gridGradient: [20, 80] },
  { name: 'left', grid: [-7 / 8, -3 / 8], output: 30, featureGradient: [[0, 3 / 16], [4, 9 / 16]], gridGradient: [80, 60] },
  { name: 'top', grid: [-3 / 8, -7 / 8], output: 13.125, featureGradient: [[0, 3 / 16], [1, 9 / 16]], gridGradient: [15, 35] },
  { name: 'right', grid: [7 / 8, -3 / 8], output: 52.5, featureGradient: [[3, 3 / 16], [7, 9 / 16]], gridGradient: [-140, 60] },
  { name: 'bottom', grid: [-3 / 8, 7 / 8], output: 103.125, featureGradient: [[12, 3 / 16], [13, 9 / 16]], gridGradient: [15, -275] },
  { name: 'top-left', grid: [-7 / 8, -7 / 8], output: 5.625, featureGradient: [[0, 9 / 16]], gridGradient: [15, 15] },
];

function valueInfo(name, shape) {
  return {
    name,
    type: {
      value: {
        case: 'tensorType',
        value: { elemType: 1, shape: { dim: shape.map((dimension) => ({ value: { case: 'dimValue', value: BigInt(dimension) } })) } },
      },
    },
  };
}

const modelProto = create(ModelProtoSchema, {
  irVersion: 9n,
  producerName: 'Open Vision Builder operator probe',
  opsetImport: [{ domain: '', version: 17n }],
  graph: {
    name: 'grid-sample-bilinear-zeros',
    input: [valueInfo('features', [1, 1, 4, 4]), valueInfo('grid', [1, 1, 1, 2])],
    output: [valueInfo('output', [1, 1, 1, 1])],
    node: [{
      opType: 'GridSample', input: ['features', 'grid'], output: ['output'],
      attribute: [
        { name: 'mode', type: AttributeProto_AttributeType.STRING, s: new TextEncoder().encode('bilinear') },
        { name: 'padding_mode', type: AttributeProto_AttributeType.STRING, s: new TextEncoder().encode('zeros') },
        { name: 'align_corners', type: AttributeProto_AttributeType.INT, i: 0n },
      ],
    }],
  },
});
const bytes = toBinary(ModelProtoSchema, modelProto);

function maxAbsoluteError(actual, expected) {
  if (actual.length !== expected.length) return Infinity;
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) return Infinity;
    maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
  }
  return maximum;
}

defaultDevice('wasm');
const model = new ONNXModel(bytes);
const differentiate = grad((inputs) => model.run(inputs).output.sum());
const results = [];
try {
  for (const fixture of fixtures) {
    const inputs = {
      features: np.array(featureValues, { shape: [1, 1, 4, 4], dtype: np.float32 }),
      grid: np.array(new Float32Array(fixture.grid), { shape: [1, 1, 1, 2], dtype: np.float32 }),
    };
    const [actualOutput] = await model.run(tree.ref(inputs)).output.data();
    const gradients = differentiate(inputs);
    const [actualFeatures, actualGrid] = await Promise.all([
      gradients.features.data(), gradients.grid.data(),
    ]);
    const expectedFeatures = new Float32Array(16);
    for (const [index, weight] of fixture.featureGradient) expectedFeatures[index] = weight;
    const errors = {
      output: maxAbsoluteError([actualOutput], [fixture.output]),
      featureGradient: maxAbsoluteError(actualFeatures, expectedFeatures),
      gridGradient: maxAbsoluteError(actualGrid, fixture.gridGradient),
    };
    results.push({
      name: fixture.name,
      grid: fixture.grid,
      expected: { output: fixture.output, featureGradient: Array.from(expectedFeatures), gridGradient: fixture.gridGradient },
      actual: { output: actualOutput, featureGradient: Array.from(actualFeatures), gridGradient: Array.from(actualGrid) },
      maxAbsoluteError: errors,
      passed: Object.entries(errors).every(([name, error]) => error <= tolerance[name]),
    });
  }
} finally {
  model.dispose();
}

const passed = results.every((result) => result.passed);
console.log(JSON.stringify({
  scope: 'GridSample operator prerequisite only; no complete D-FINE model was loaded or trained.',
  operatorGate: passed ? 'PASS' : 'FAIL',
  detectorGates: 'NOT RUN',
  runtime: { node: process.version, platform: process.platform, arch: process.arch, backend: 'wasm' },
  versions,
  model: { ir: 9, opset: 17, sha256: createHash('sha256').update(bytes).digest('hex') },
  tolerance,
  results,
}, null, 2));
process.exitCode = passed ? 0 : 1;
