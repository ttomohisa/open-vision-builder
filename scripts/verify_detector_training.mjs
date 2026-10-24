// Developer/CI qualification on a pinned public reference, independent of the app.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';

const root = fileURLToPath(new URL('../', import.meta.url));
const { values: args } = parseArgs({ options: {
  reference: { type: 'string', default: resolve(root, 'artifacts/p1b/training/pretrained-reference.json') },
  report: { type: 'string', default: resolve(root, 'artifacts/p1b/training/pretrained-js-verification.json') },
} });
const reference = JSON.parse(readFileSync(args.reference, 'utf8'));
if (reference.schemaVersion !== 1 || reference.scope !== 'pretrained-final-head'
  || reference.source?.sha256 !== '19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b') {
  throw new Error('This qualification requires the independent reference from the exact pinned pretrained model.');
}
const tolerance = JSON.parse(readFileSync(new URL('../tests/fixtures/detector-training-tolerances.json', import.meta.url), 'utf8'));
// Transform browser-compatible TypeScript without opening HTTP or WebSocket listeners.
const server = await createServer({
  configFile: false, root, server: { middlewareMode: true, hmr: false, ws: false }, appType: 'custom',
});
const { computeDFineHeadGradients, trainDFineHead, DFINE_TRAINABLE_NAMES } = await server.ssrLoadModule('/src/ml/detection/head.ts');
const config = { ...reference.config, projection: new Float32Array(reference.config.projection) };
const features = Object.fromEntries(Object.entries(reference.features).map(([key, tensor]) => [key, new Float32Array(tensor.data)]));
const initial = Object.fromEntries(Object.entries(reference.weights).map(([key, tensor]) => [key, {
  shape: tensor.shape, data: new Float32Array(tensor.data),
}]));
const lossNames = ['total', 'vfl', 'l1', 'giou'];
const outputNames = ['logits', 'boxes', 'corners'];
const targets = (source) => ({ labels: new Int32Array(source.targets.labels), boxes: new Float32Array(source.targets.boxes) });

function compare(actual, expected, bound) {
  let maxAbsolute = 0; let maxRelative = 0; let failures = 0;
  if (actual.length !== expected.length) throw new Error('Shape mismatch');
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    maxAbsolute = Math.max(maxAbsolute, error);
    maxRelative = Math.max(maxRelative, error / Math.max(Math.abs(expected[i]), 1e-12));
    if (!Number.isFinite(actual[i]) || error > bound.absolute + bound.relative * Math.abs(expected[i])) failures++;
  }
  return { values: actual.length, failures, maxAbsolute, maxRelative, passed: failures === 0 };
}

function changed(actual, before) {
  let count = 0; let maxAbsolute = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== before[i]) count++;
    maxAbsolute = Math.max(maxAbsolute, Math.abs(actual[i] - before[i]));
  }
  return { values: actual.length, count, maxAbsolute };
}

function writeReport(result) {
  mkdirSync(dirname(args.report), { recursive: true });
  writeFileSync(args.report, JSON.stringify(result, null, 2) + '\n');
}

const result = {
  scope: 'Real COCO-pretrained final D-FINE localization/class head; fixed prefix/LQE; Node WASM',
  source: reference.source,
  runtime: { node: process.version, platform: process.platform, architecture: process.arch },
  tolerance, sourceParity: reference.sourceParity, cases: {}, updates: {},
};
let initialPrediction;
try {
  for (const name of ['empty', 'single', 'multiple']) {
    const source = reference.cases[name];
    const started = performance.now();
    const actual = await computeDFineHeadGradients(features, targets(source), initial, config);
    const output = {
      milliseconds: performance.now() - started,
      prediction: {}, gradients: {},
      matches: JSON.stringify(Array.from(actual.matches.predictionIndices)) === JSON.stringify(source.matches.predictionIndices)
        && JSON.stringify(Array.from(actual.matches.targetIndices)) === JSON.stringify(source.matches.targetIndices),
    };
    if (name === 'multiple') initialPrediction = actual.prediction;
    for (const key of outputNames) output.prediction[key] = compare(actual.prediction[key], source.prediction[key].data, tolerance.pretrainedForward);
    output.loss = compare(lossNames.map((key) => actual.losses[key]), lossNames.map((key) => source.losses[key]), tolerance.pretrainedLoss);
    for (const key of DFINE_TRAINABLE_NAMES) output.gradients[key] = compare(actual.gradients[key].data, source.gradients[key].data, tolerance.pretrainedGradient);
    result.cases[name] = output;
    console.log(name, JSON.stringify({ milliseconds: output.milliseconds, matches: output.matches, loss: output.loss, prediction: output.prediction }));
  }

  const source = reference.cases.multiple;
  const trained = await trainDFineHead(features, targets(source), initial, config, { steps: 2, learningRate: 1e-4 });
  const after = await computeDFineHeadGradients(features, targets(source), trained.weights, config);
  result.updates = {
    metrics: trained.metrics,
    preUpdateLosses: trained.losses,
    after: after.losses,
    parameters: {}, parameterChanges: {}, prediction: {},
    boxChanges: changed(after.prediction.boxes, initialPrediction.boxes),
    frozenBitIdentical: true,
    lossDecreased: after.losses.total < trained.losses[0],
  };
  for (const key of DFINE_TRAINABLE_NAMES) {
    result.updates.parameters[key] = compare(trained.weights[key].data, source.steps[1].weights[key].data, tolerance.pretrainedSgdWeights);
    result.updates.parameterChanges[key] = changed(trained.weights[key].data, initial[key].data);
  }
  for (const key of Object.keys(initial).filter((name) => !DFINE_TRAINABLE_NAMES.includes(name))) {
    result.updates.frozenBitIdentical &&= initial[key].data.every((value, i) => Object.is(value, trained.weights[key].data[i]));
  }
  for (const key of outputNames) result.updates.prediction[key] = compare(after.prediction[key], source.steps[1].prediction[key].data, tolerance.pretrainedForward);

  const gates = Object.values(result.cases).flatMap((item) => [
    item.matches, item.loss.passed,
    ...Object.values(item.prediction).map((value) => value.passed),
    ...Object.values(item.gradients).map((value) => value.passed),
  ]);
  gates.push(
    ...Object.values(result.updates.parameters).map((value) => value.passed),
    ...Object.values(result.updates.prediction).map((value) => value.passed),
    ...Object.values(result.updates.parameterChanges).map((value) => value.count > 0),
    result.updates.frozenBitIdentical, result.updates.lossDecreased, result.updates.boxChanges.count > 0,
    trained.metrics.localizationGradientL2 > 0, trained.metrics.classificationGradientL2 > 0,
  );
  result.passed = gates.every(Boolean);
  writeReport(result);
  console.log(JSON.stringify({ passed: result.passed, updates: result.updates }, null, 2));
  process.exitCode = result.passed ? 0 : 1;
} catch (error) {
  result.passed = false;
  result.error = String(error);
  writeReport(result);
  throw error;
} finally {
  await server.close();
}
