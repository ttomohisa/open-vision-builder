// Developer qualification harness. This is deliberately outside the product UI.
import * as ort from 'onnxruntime-web/wasm';
import { strToU8, zipSync } from 'fflate';
import {
  computeDFineHeadGradients, predictDFineHead, trainDFineHead,
  DFINE_FROZEN_NAMES, DFINE_TRAINABLE_NAMES,
  type DFineHeadConfig, type DFineHeadFeatures, type DFineWeights,
} from '../../src/ml/detection/head';
import { exportDFineOnnx, type DFineExportRecipe } from '../../src/ml/detection/export';

type Artifact = { file: string; bytes: number; sha256: string; shape?: number[] };
type Manifest = {
  schemaVersion: number; status: string; passed: boolean;
  source: { modelId: string; revision: string; sha256: string; license: string };
  artifacts: Record<string, Artifact>;
  exportRecipe: DFineExportRecipe; labels: string[]; licenses: string[];
  fixtures: { phase: number; inputSha256: string; outputs: Record<string, Artifact>;
    comparisons: Record<string, { passed: boolean; [key: string]: unknown }> }[];
};

// Fixed before browser execution. The Python verifier independently pins these.
const tolerance = {
  logits: { atol: 2e-4, rtol: 1e-4 }, boxes: { atol: 2e-5, rtol: 1e-4 },
  bboxInput: { atol: 2e-4, rtol: 1e-4 }, classInput: { atol: 2e-4, rtol: 1e-4 },
  residualCorners: { atol: 2e-4, rtol: 1e-4 }, references: { atol: 2e-5, rtol: 1e-4 },
};
const outputNames = ['logits', 'boxes'] as const;
const featureNames = ['bboxInput', 'classInput', 'residualCorners', 'references'] as const;
const tensorShapes = {
  logits: [1, 300, 80], boxes: [1, 300, 4], bboxInput: [1, 300, 128],
  classInput: [1, 300, 128], residualCorners: [1, 300, 132], references: [1, 300, 4],
} as const;

/** Check the complete preparation evidence before any model or fixture is used.
 * This pure metadata boundary also protects local browser runs outside CI.
 */
export function validateDetectorManifest(value: unknown): Manifest {
  const fail = (detail: string): never => { throw new Error(`Invalid detector manifest: ${detail}`); };
  const record = (entry: unknown, context: string): Record<string, unknown> => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return fail(context);
    return entry as Record<string, unknown>;
  };
  const keys = (entry: Record<string, unknown>, expected: readonly string[], context: string) => {
    if (Object.keys(entry).length !== expected.length || expected.some(name => !Object.hasOwn(entry, name))) fail(context);
  };
  const shape = (entry: unknown, expected: readonly number[], context: string) => {
    if (!Array.isArray(entry) || entry.length !== expected.length || entry.some((n, i) => n !== expected[i])) fail(`${context} shape`);
  };
  const digest = (entry: unknown, context: string) => {
    if (typeof entry !== 'string' || !/^[a-f0-9]{64}$/.test(entry)) fail(`${context} digest`);
  };
  const finite = (entry: unknown, context: string): number => {
    if (typeof entry !== 'number' || !Number.isFinite(entry) || entry < 0) return fail(context);
    return entry;
  };
  const artifact = (entry: unknown, file: string, context: string, expectedShape?: readonly number[]) => {
    const spec = record(entry, `${context} artifact`);
    if (spec.file !== file || !Number.isSafeInteger(spec.bytes) || Number(spec.bytes) < 1 || Number(spec.bytes) > 25 * 1024 * 1024) {
      fail(`${context} artifact path/bytes`);
    }
    digest(spec.sha256, `${context} artifact`);
    if (expectedShape) {
      shape(spec.shape, expectedShape, context);
      if (spec.dtype !== 'float32-le' || spec.bytes !== expectedShape.reduce((count, n) => count * n, 4)) fail(`${context} dtype/bytes`);
    }
  };
  const limits = (entry: unknown, name: keyof typeof tolerance) => {
    const actual = record(entry, `${name} tolerance`), expected = tolerance[name];
    if (actual.atol !== expected.atol || actual.rtol !== expected.rtol) fail(`${name} tolerance changed`);
  };

  const manifest = record(value, 'metadata');
  if (manifest.schemaVersion !== 1 || manifest.id !== 'dfine-nano' || manifest.task !== 'object-detection' || typeof manifest.passed !== 'boolean') {
    fail('identity or qualification metadata');
  }
  const source = record(manifest.source, 'source');
  if (source.modelId !== 'ustc-community/dfine-nano-coco' || source.revision !== '066438d3d8f0da137a37b38fdf3368fd4afceced' ||
      source.sha256 !== '19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b' || source.license !== 'Apache-2.0') fail('source');
  if (!Array.isArray(manifest.labels) || manifest.labels.length !== 80 || manifest.labels.some(label => typeof label !== 'string' || !label.trim())) fail('labels');
  const licenseNames = ['LICENSE.txt', 'TRANSFORMERS_LICENSE.txt', 'NOTICE.txt', 'MODEL_CARD.md'];
  if (!Array.isArray(manifest.licenses) || manifest.licenses.length !== licenseNames.length ||
      licenseNames.some(name => !(manifest.licenses as unknown[]).includes(name))) fail('licenses');
  const input = record(manifest.input, 'input');
  if (input.name !== 'images' || input.dtype !== 'float32') fail('input');
  shape(input.shape, [1, 3, 640, 640], 'input');
  const outputs = record(manifest.outputs, 'outputs');
  keys(outputs, outputNames, 'outputs');
  for (const name of outputNames) shape(record(outputs[name], name).shape, tensorShapes[name], name);
  const artifacts = record(manifest.artifacts, 'artifacts');
  keys(artifacts, ['detector.onnx', 'frozen-prefix.onnx', 'headWeights'], 'artifacts');
  artifact(artifacts['detector.onnx'], 'detector.onnx', 'model');
  artifact(artifacts['frozen-prefix.onnx'], 'frozen-prefix.onnx', 'prefix');
  artifact(artifacts.headWeights, 'head-weights.json', 'head');

  const recipe = record(manifest.exportRecipe, 'export recipe');
  if (recipe.schemaVersion !== 1 || recipe.id !== 'dfine-nano-coco-final-head-v1') fail('export recipe');
  const recipeInput = record(recipe.input, 'export input');
  if (recipeInput.name !== 'images') fail('export input');
  shape(recipeInput.shape, [1, 3, 640, 640], 'export input');
  if (!Array.isArray(recipe.outputs) || recipe.outputs.length !== 2) fail('export outputs');
  for (const [i, name] of outputNames.entries()) {
    const output = record((recipe.outputs as unknown[])[i], 'export output');
    if (output.name !== name) fail('export output');
    shape(output.shape, tensorShapes[name], 'export output');
  }
  if (!Array.isArray(recipe.parameters) || recipe.parameters.length !== DFINE_TRAINABLE_NAMES.length) fail('export parameters');
  const parameterNames = (recipe.parameters as unknown[]).map(entry => record(entry, 'export parameter').parameter);
  if (new Set(parameterNames).size !== DFINE_TRAINABLE_NAMES.length || DFINE_TRAINABLE_NAMES.some(name => !parameterNames.includes(name))) fail('export parameters');

  const names = [...outputNames, ...featureNames], phases = [0, 37, 91];
  const declaredLimits = record(manifest.tolerances, 'tolerances');
  keys(declaredLimits, names, 'tolerances');
  for (const name of names) limits(declaredLimits[name], name);
  if (!Array.isArray(manifest.fixtures) || manifest.fixtures.length !== phases.length) fail('fixtures');
  let allPassed = true;
  for (const [i, entry] of (manifest.fixtures as unknown[]).entries()) {
    const fixture = record(entry, 'fixture'), phase = phases[i];
    if (fixture.phase !== phase || fixture.id !== `dfine-rgb-pattern-v1-phase-${phase}`) fail('fixture phases');
    digest(fixture.inputSha256, 'fixture input');
    const fixtureOutputs = record(fixture.outputs, 'fixture outputs'), comparisons = record(fixture.comparisons, 'fixture comparisons');
    keys(fixtureOutputs, names, 'fixture outputs');
    keys(comparisons, names, 'fixture comparisons');
    for (const name of names) {
      artifact(fixtureOutputs[name], `fixtures/pattern-${phase}/${name}.f32`, `${phase}/${name} output`, tensorShapes[name]);
      const row = record(comparisons[name], `${phase}/${name} comparison`);
      shape(row.shape, tensorShapes[name], `${phase}/${name} comparison`);
      limits(row, name);
      finite(row.maxAbsoluteError, `${phase}/${name} comparison absolute error`);
      const ratio = finite(row.maxToleranceRatio, `${phase}/${name} comparison tolerance ratio`);
      if (typeof row.passed !== 'boolean' || row.passed !== (ratio <= 1)) fail(`${phase}/${name} comparison result`);
      if (['logits', 'boxes', 'references'].includes(name) && !row.passed) fail(`${phase}/${name} final-output/reference failure`);
      allPassed &&= row.passed === true;
    }
  }
  if (manifest.passed !== allPassed || manifest.status !== (allPassed ? 'developer-investigation' : 'unqualified-intermediate-parity-failure')) {
    fail('qualification status contradicts completed comparisons');
  }
  return value as Manifest;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer);
  return [...new Uint8Array(digest)].map(n => n.toString(16).padStart(2, '0')).join('');
}

function compare(actual: Float32Array, expected: Float32Array, limits: { atol: number; rtol: number }) {
  if (actual.length !== expected.length || actual.length === 0) throw new Error('Parity tensor shape mismatch');
  let maximumAbsoluteError = 0, maximumToleranceRatio = 0, failures = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) throw new Error('Non-finite parity value');
    const error = Math.abs(actual[i] - expected[i]);
    const ratio = error / (limits.atol + limits.rtol * Math.abs(expected[i]));
    maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
    maximumToleranceRatio = Math.max(maximumToleranceRatio, ratio);
    if (ratio > 1) failures++;
  }
  return { passed: failures === 0, count: actual.length, failures, maximumAbsoluteError, maximumToleranceRatio };
}

function pattern(phase: number): Float32Array {
  const size = 640, plane = size * size, result = new Float32Array(3 * plane);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const offset = y * size + x;
    result[offset] = ((7 * x + 3 * y + phase) % 256) / 255;
    result[plane + offset] = ((2 * x + 11 * y + 3 * phase) % 256) / 255;
    result[2 * plane + offset] = ((13 * x + 5 * y + 7 * phase) % 256) / 255;
  }
  return result;
}

async function infer(bytes: Uint8Array, input: Float32Array) {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);
  let outputs: ort.InferenceSession.OnnxValueMapType | undefined;
  try {
    outputs = await session.run({ images: tensor });
    return Object.fromEntries(Object.entries(outputs).map(([name, value]) => {
      if (!(value.data instanceof Float32Array)) throw new Error(`Non-FP32 output: ${name}`);
      return [name, new Float32Array(value.data)];
    }));
  } finally {
    tensor.dispose();
    if (outputs) Object.values(outputs).forEach(output => output.dispose());
    await session.release();
  }
}

export async function qualifyDetector(baseUrl: string) {
  const started = performance.now();
  const base = new URL(baseUrl), modelRoot = new URL('models/dfine-nano/', base);
  const stage = (message: string) => { document.body.textContent = message; };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL('runtime/onnxruntime/', base).href;
  async function fetchBytes(url: URL) {
    if (url.origin !== location.origin || !url.href.startsWith(base.href)) throw new Error('Unexpected artifact origin or path');
    const response = await fetch(url, { credentials: 'omit' });
    if (!response.ok) throw new Error(`${url.pathname}: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  }
  async function artifact(spec: Artifact) {
    if (!/^[a-zA-Z0-9_.\-/]+$/.test(spec.file) || spec.file.split('/').includes('..')) throw new Error('Invalid artifact path');
    const bytes = await fetchBytes(new URL(spec.file, modelRoot));
    if (bytes.length !== spec.bytes || await sha256(bytes) !== spec.sha256) throw new Error(`Artifact digest mismatch: ${spec.file}`);
    return bytes;
  }
  stage('Checking the pinned public D-FINE artifacts');
  const manifestBytes = await fetchBytes(new URL('model.json', modelRoot));
  const manifest = validateDetectorManifest(JSON.parse(new TextDecoder().decode(manifestBytes)));
  const [model, prefix, headBytes] = await Promise.all([
    artifact(manifest.artifacts['detector.onnx']), artifact(manifest.artifacts['frozen-prefix.onnx']), artifact(manifest.artifacts.headWeights),
  ]);
  const savedHead = JSON.parse(new TextDecoder().decode(headBytes)) as {
    schemaVersion: number; config: Omit<DFineHeadConfig, 'projection'> & { projection: number[] };
    weights: Record<string, { shape: number[]; data: number[] }>;
  };
  if (savedHead.schemaVersion !== 1) throw new Error('Unexpected head schema');
  const config = { ...savedHead.config, projection: new Float32Array(savedHead.config.projection) };
  const weights: DFineWeights = Object.fromEntries(Object.entries(savedHead.weights)
    .map(([name, value]) => [name, { shape: value.shape, data: new Float32Array(value.data) }]));
  const fixture = manifest.fixtures.find(f => f.phase === 0);
  if (!fixture) throw new Error('Missing independent public fixture');
  const input = pattern(0), inputSha256 = await sha256(new Uint8Array(input.buffer));
  if (inputSha256 !== fixture.inputSha256) throw new Error('Procedural input differs from the independent source');
  const native: Record<string, Float32Array> = {};
  for (const name of [...outputNames, ...featureNames]) {
    const bytes = await artifact(fixture.outputs[name]);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    native[name] = Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
  }
  stage('Running the complete detector and the frozen prefix in ORT Web');
  const original = await infer(model, input), features = await infer(prefix, input) as unknown as DFineHeadFeatures;
  const nativeComparison = Object.fromEntries([...outputNames, ...featureNames].map(name =>
    [name, compare(name in original ? original[name] : features[name as keyof DFineHeadFeatures], native[name], tolerance[name])]));
  const initial = await predictDFineHead(features, weights, config);
  const headComparison = Object.fromEntries(outputNames.map(name => [name, compare(initial[name], original[name], tolerance[name])]));
  stage('Updating pretrained localization and classification weights in browser WASM');
  const targets = { labels: new Int32Array([0, 15]), boxes: new Float32Array([0.2, 0.3, 0.25, 0.35, 0.7, 0.65, 0.2, 0.3]) };
  const result = await trainDFineHead(features, targets, weights, config, { steps: 2, learningRate: 1e-4 });
  const final = await computeDFineHeadGradients(features, targets, result.weights, config);
  const frozenUnchanged = DFINE_FROZEN_NAMES.every(name => result.weights[name].data.every((n, i) => Object.is(n, weights[name].data[i])));
  const outputChange = Object.fromEntries(outputNames.map(name => [name,
    Math.max(...final.prediction[name].map((n, i) => Math.abs(n - initial[name][i])))]));
  stage('Generating and reopening the complete ONNX model');
  const trained = Object.fromEntries(DFINE_TRAINABLE_NAMES.map(name => [name, result.weights[name]]));
  const exported = exportDFineOnnx(model, manifest.exportRecipe, trained);
  const fresh = await infer(exported, input);
  const exportComparison = Object.fromEntries(outputNames.map(name => [name, compare(fresh[name], final.prediction[name], tolerance[name])]));
  const preparationFailures = manifest.fixtures.flatMap(f => Object.entries(f.comparisons)
    .filter(([, result]) => !result.passed).map(([tensor]) => ({ phase: f.phase, tensor })));
  const browserFailures = Object.entries(nativeComparison).filter(([, result]) => !result.passed).map(([tensor]) => ({ phase: 0, tensor }));
  const qualification = {
    status: manifest.passed && browserFailures.length === 0 ? 'PASS' : 'FAIL',
    scope: 'Native detector forward including frozen intermediate features; unchanged elementwise gates.',
    preparationStatus: manifest.status, preparationFailures, browserFailures,
    releaseReady: false,
  };
  const report = {
    schemaVersion: 1,
    scope: 'Developer qualification: pretrained final bbox/class head SGD, frozen prefix and LQE, original 80 COCO classes; no quality or complete fine-tuning claim.',
    source: manifest.source, preparedStatus: manifest.status, detectorQualification: qualification,
    preparedArtifacts: { manifestSha256: await sha256(manifestBytes), files: manifest.artifacts },
    artifact: { file: 'model.onnx', sha256: await sha256(exported), bytes: exported.length },
    training: {
      recipe: manifest.exportRecipe.id, backend: result.backend, stepsCompleted: result.stepsCompleted,
      learningRate: 1e-4, lossesBeforeUpdate: result.losses, lossAfterTraining: final.losses.total,
      metrics: result.metrics, frozenLqeUnchanged: frozenUnchanged, outputChange,
      trainableParameters: [...DFINE_TRAINABLE_NAMES],
      targets: { labels: [...targets.labels], boxes: [...targets.boxes], source: 'Procedural qualification only, not real annotations or accuracy evaluation.' },
    },
    verification: {
      tolerances: { logits: tolerance.logits, boxes: tolerance.boxes },
      nativeFeatureTolerances: Object.fromEntries(featureNames.map(name => [name, tolerance[name]])),
      native: nativeComparison, originalHead: headComparison, export: exportComparison,
    },
    parityFixture: { name: 'dfine-rgb-pattern-v1', phase: 0, shape: [1, 3, 640, 640], inputSha256,
      expectedLogits: [...final.prediction.logits], expectedBoxes: [...final.prediction.boxes] },
    environment: { userAgent: navigator.userAgent, crossOriginIsolated, threads: 1, graphOptimizationLevel: 'all',
      durationMs: performance.now() - started, durationScope: 'Preparation, browser inference, two head updates and export parity; archive creation excluded.' },
  };
  const kitManifest = {
    schemaVersion: 1, task: 'object-detection', recipe: manifest.exportRecipe.id, source: manifest.source,
    detectorQualification: qualification, intendedUse: 'Developer investigation only; not a qualified product model.',
    model: report.artifact, labels: 'labels.json', input: { name: 'images', shape: [1, 3, 640, 640],
      color: 'RGB', scale: 'divide by 255', resize: 'bilinear half-pixel full-image stretch', alpha: 'composite-white-float', exif: 'apply', icc: 'disabled' },
    outputs: { logits: { name: 'logits', shape: [1, 300, 80] }, boxes: { name: 'boxes', shape: [1, 300, 4], format: 'normalized cxcywh' } },
    postprocess: { scores: 'sigmoid', selection: 'global query/class pairs', nms: false, clipBoxes: false },
  };
  const files: Record<string, Uint8Array> = {
    'model.onnx': exported, 'report.json': strToU8(JSON.stringify(report, null, 2)),
    'labels.json': strToU8(JSON.stringify(manifest.labels, null, 2)), 'manifest.json': strToU8(JSON.stringify(kitManifest, null, 2)),
  };
  for (const name of ['LICENSE.txt', 'TRANSFORMERS_LICENSE.txt', 'NOTICE.txt', 'MODEL_CARD.md']) {
    files[name] = await fetchBytes(new URL(name, modelRoot));
  }
  files['NOTICE.txt'] = strToU8(new TextDecoder().decode(files['NOTICE.txt']) +
    '\nBrowser-generated derivative: two SGD updates, learning rate 0.0001, recipe ' + manifest.exportRecipe.id +
    '. Updated the eight final bbox/class tensors listed in report.json; all other weights retained.\n' +
    `Detector qualification: ${qualification.status}. This is an investigation artifact; no quality or product-readiness claim.\n`);
  files['detect.py'] = await fetchBytes(new URL('kit/detect.py', base));
  files['README.txt'] = await fetchBytes(new URL('kit/DETECTION_README.txt', base));
  const archive = zipSync(files, { level: 1 });
  const url = URL.createObjectURL(new Blob([new Uint8Array(archive)], { type: 'application/zip' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'dfine-qualification-kit.zip';
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  stage(`Component experiment completed. Detector forward qualification: ${qualification.status}. See report.json.`);
  return { report, kitFiles: Object.keys(files) };
}
