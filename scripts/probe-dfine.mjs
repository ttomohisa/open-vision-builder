// A prerequisite operator probe, not a D-FINE model or training benchmark.
// Exit 1 means forward/backward disagrees with an independent reference.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { create, toBinary } from '@bufbuild/protobuf';
import { AttributeProto_AttributeType, ModelProtoSchema } from 'onnx-buf';

const require = createRequire(import.meta.url);
const root = dirname(dirname(fileURLToPath(import.meta.url)));
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

// Frozen hand-derived fixtures; their coordinates avoid integer-pixel knots.
const tolerance = { output: 1e-5, featureGradient: 1e-5, gridGradient: 1e-5 };
// Frozen before the expanded reference suite was run; do not tune to results.
// Per element: abs(actual - expected) <= atol + rtol * abs(expected).
const pytorchTolerance = {
  output: { atol: 1e-5, rtol: 1e-5 },
  featureGradient: { atol: 2e-5, rtol: 1e-5 },
  gridGradient: { atol: 2e-5, rtol: 1e-5 },
};
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

function modelBytes(featureShape, gridShape, alignCorners) {
  const modelProto = create(ModelProtoSchema, {
    irVersion: 9n,
    producerName: 'Open Vision Builder operator probe',
    opsetImport: [{ domain: '', version: 17n }],
    graph: {
      name: 'grid-sample-bilinear-zeros',
      input: [valueInfo('features', featureShape), valueInfo('grid', gridShape)],
      output: [valueInfo('output', [featureShape[0], featureShape[1], gridShape[1], gridShape[2]])],
      node: [{
        opType: 'GridSample', input: ['features', 'grid'], output: ['output'],
        attribute: [
          { name: 'mode', type: AttributeProto_AttributeType.STRING, s: new TextEncoder().encode('bilinear') },
          { name: 'padding_mode', type: AttributeProto_AttributeType.STRING, s: new TextEncoder().encode('zeros') },
          { name: 'align_corners', type: AttributeProto_AttributeType.INT, i: BigInt(alignCorners) },
        ],
      }],
    },
  });
  return toBinary(ModelProtoSchema, modelProto);
}
const bytes = modelBytes([1, 1, 4, 4], [1, 1, 1, 2], 0);

function maxAbsoluteError(actual, expected) {
  if (actual.length !== expected.length) return Infinity;
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) return Infinity;
    maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
  }
  return maximum;
}

export async function runGridSampleProbe({ format = 'esm' } = {}) {
  if (!['esm', 'cjs'].includes(format)) throw new Error(`Unsupported module format: ${format}`);
  // CJS must use its matching jax-js instance, not mix ESM and CJS tensor classes.
  const { defaultDevice, grad, numpy: np, tree } = format === 'cjs'
    ? require('@jax-js/jax') : await import('@jax-js/jax');
  const { ONNXModel } = format === 'cjs' ? require('@jax-js/onnx') : await import('@jax-js/onnx');
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

  const referencePath = resolve(root, 'tests/fixtures/grid-sample-pytorch.json');
  const referenceBytes = readFileSync(referencePath);
  const reference = JSON.parse(referenceBytes);
  if (reference.schemaVersion !== 1 || reference.reference.torch !== '2.6.0+cpu'
      || reference.reference.mode !== 'bilinear' || reference.reference.paddingMode !== 'zeros'
      || reference.reference.dtype !== 'float32' || reference.reference.device !== 'cpu'
      || reference.cases.length !== 12) {
    throw new Error('Unrecognized PyTorch GridSample reference; review before updating this gate.');
  }

  function compare(actual, expected, limits) {
    if (actual.length !== expected.length) throw new Error('GridSample reference length mismatch');
    let maximumAbsoluteError = 0;
    let maximumToleranceRatio = 0;
    let worstIndex = -1;
    for (let i = 0; i < actual.length; i++) {
      if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) {
        return { passed: false, maximumAbsoluteError: null, maximumToleranceRatio: null, worstIndex: i, nonFinite: true };
      }
      const error = Math.abs(actual[i] - expected[i]);
      const ratio = error / (limits.atol + limits.rtol * Math.abs(expected[i]));
      maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
      if (ratio > maximumToleranceRatio) { maximumToleranceRatio = ratio; worstIndex = i; }
    }
    return { passed: maximumToleranceRatio <= 1, maximumAbsoluteError, maximumToleranceRatio, worstIndex };
  }

  const pytorchResults = [];
  for (const fixture of reference.cases) {
    const fixtureBytes = modelBytes(fixture.featureShape, fixture.gridShape, fixture.alignCorners);
    const fixtureModel = new ONNXModel(fixtureBytes);
    try {
      const inputs = {
        features: np.array(new Float32Array(fixture.features), { shape: fixture.featureShape, dtype: np.float32 }),
        grid: np.array(new Float32Array(fixture.grid), { shape: fixture.gridShape, dtype: np.float32 }),
      };
      const output = fixtureModel.run(tree.ref(inputs)).output;
      const outputShape = output.shape;
      const actualOutput = await output.data();
      const gradients = grad((operands) => fixtureModel.run(operands).output.mul(
        np.array(new Float32Array(fixture.cotangent), { shape: fixture.outputShape, dtype: np.float32 }),
      ).sum())(inputs);
      const [actualFeatures, actualGrid] = await Promise.all([
        gradients.features.data(), gradients.grid.data(),
      ]);
      const actual = { output: Array.from(actualOutput), featureGradient: Array.from(actualFeatures), gridGradient: Array.from(actualGrid) };
      const comparison = Object.fromEntries(Object.entries(actual).map(([name, values]) =>
        [name, compare(values, fixture.expected[name], pytorchTolerance[name])],
      ));
      pytorchResults.push({
        name: fixture.name, alignCorners: fixture.alignCorners,
        featureShape: fixture.featureShape, gridShape: fixture.gridShape, outputShape,
        modelSha256: createHash('sha256').update(fixtureBytes).digest('hex'),
        comparison,
        passed: JSON.stringify(outputShape) === JSON.stringify(fixture.outputShape)
          && Object.values(comparison).every((comparison) => comparison.passed),
        actual,
      });
    } finally {
      fixtureModel.dispose();
    }
  }

  const passed = results.every((result) => result.passed) && pytorchResults.every((result) => result.passed);
  const onnxEntry = format === 'cjs' ? require.resolve('@jax-js/onnx')
    : fileURLToPath(import.meta.resolve('@jax-js/onnx'));
  return {
    scope: 'GridSample operator prerequisite only; no complete D-FINE model was loaded or trained.',
    operatorGate: passed ? 'PASS' : 'FAIL',
    detectorGates: 'NOT RUN',
    runtime: { node: process.version, platform: process.platform, arch: process.arch, backend: 'wasm', moduleFormat: format },
    versions,
    implementation: { package: '@jax-js/onnx', entrySha256: createHash('sha256').update(readFileSync(onnxEntry)).digest('hex') },
    model: { ir: 9, opset: 17, sha256: createHash('sha256').update(bytes).digest('hex') },
    tolerance,
    results,
    pytorch: {
      reference: reference.reference,
      fixture: 'tests/fixtures/grid-sample-pytorch.json',
      fixtureSha256: createHash('sha256').update(referenceBytes).digest('hex'),
      tolerance: pytorchTolerance,
      results: pytorchResults,
    },
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const format = args.length === 0 ? 'esm' : args[1];
  if (!['esm', 'cjs'].includes(format) || (args.length !== 0 && (args.length !== 2 || args[0] !== '--format'))) {
    throw new Error('Usage: node scripts/probe-dfine.mjs [--format esm|cjs]');
  }
  const report = await runGridSampleProbe({ format });
  console.log(JSON.stringify(report, null, 2));
  process.exitCode = report.operatorGate === 'PASS' ? 0 : 1;
}
