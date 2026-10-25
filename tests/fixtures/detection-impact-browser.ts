// Developer-only real-image experiment. No native features or Python results are loaded here.
import * as ort from 'onnxruntime-web/wasm';
import { validateDetectorManifest } from './detection-browser';
import {
  computeDFineHeadGradients, trainDFineHead, DFINE_TRAINABLE_NAMES, DFINE_FROZEN_NAMES,
  type DFineHeadConfig, type DFineHeadFeatures, type DFineWeights, type DFinePrediction,
} from '../../src/ml/detection/head';
import { exportDFineOnnx } from '../../src/ml/detection/export';
import { validateImpactPlan } from '../../scripts/detector-impact-state.mjs';

type Tensor = { shape: number[]; data: number[] };
type Artifact = { file: string; bytes: number; sha256: string; shape?: number[] };
type Sample = { id: string; split: string; input: Artifact; targets: { labels: number[]; boxes: number[] } };
const featureShapes = { bboxInput: [300, 128], classInput: [300, 128], residualCorners: [300, 132], references: [300, 4] };
const predictionShapes = { logits: [300, 80], boxes: [300, 4], corners: [300, 132] };
let root: URL, planSha256: string, datasetManifestSha256: string;
let manifest: ReturnType<typeof validateDetectorManifest>, model: Uint8Array, weights: DFineWeights, initial: DFineWeights;
let config: DFineHeadConfig, prefix: ort.InferenceSession, samples: Sample[];
const cached = new Map<string, DFineHeadFeatures>();
let step = 0, pending: { id: string; weights: DFineWeights } | undefined;

async function digest(bytes: Uint8Array) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer)), n => n.toString(16).padStart(2, '0')).join('');
}
async function bytes(file: string, expected?: { sha256: string; bytes?: number }) {
  if (!/^[a-zA-Z0-9_./-]+$/.test(file) || file.startsWith('/') || file.split('/').includes('..')) throw new Error('Invalid fixture path');
  const url = new URL(file, root);
  if (url.origin !== location.origin || !url.href.startsWith(root.href)) throw new Error('Invalid fixture origin');
  const response = await fetch(url, { credentials: 'omit' });
  if (!response.ok) throw new Error(`Fixture HTTP ${response.status}: ${file}`);
  const result = new Uint8Array(await response.arrayBuffer());
  if (expected && ((expected.bytes !== undefined && expected.bytes !== result.length) || await digest(result) !== expected.sha256)) {
    throw new Error(`Fixture digest/size mismatch: ${file}`);
  }
  return result;
}
function serialWeights(value: DFineWeights): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(value).map(([name, tensor]) => [name, { shape: [...tensor.shape], data: Array.from(tensor.data) }]));
}
function serialPrediction(value: DFinePrediction): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(predictionShapes).map(([name, shape]) => [name, { shape, data: Array.from(value[name as keyof DFinePrediction]) }]));
}
function sample(id: string) {
  const found = samples.find(item => item.id === id);
  if (!found) throw new Error('Unknown image');
  return found;
}
async function input(spec: Artifact) {
  if (JSON.stringify(spec.shape) !== '[1,3,640,640]' || spec.bytes !== 4915200) throw new Error('Invalid canonical image shape');
  const data = await bytes(`data/${spec.file}`, spec);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const array = Float32Array.from({ length: data.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
  if (!array.every(n => Number.isFinite(n) && n >= 0 && n <= 1)) throw new Error('Invalid canonical image value');
  return array;
}
async function infer(session: ort.InferenceSession, data: Float32Array, shapes: Record<string, number[]>) {
  const tensor = new ort.Tensor('float32', data, [1, 3, 640, 640]);
  let output: ort.InferenceSession.OnnxValueMapType | undefined;
  try {
    output = await session.run({ images: tensor });
    if (Object.keys(output).length !== Object.keys(shapes).length) throw new Error('Unexpected output count');
    return Object.fromEntries(Object.entries(shapes).map(([name, shape]) => {
      const value = output![name];
      if (!value || !(value.data instanceof Float32Array) || JSON.stringify(value.dims) !== JSON.stringify([1, ...shape]) ||
          !value.data.every(Number.isFinite)) throw new Error(`Invalid model output: ${name}`);
      return [name, new Float32Array(value.data)];
    }));
  } finally { tensor.dispose(); if (output) Object.values(output).forEach(value => value.dispose()); }
}
async function features(id: string): Promise<DFineHeadFeatures> {
  if (!cached.has(id)) cached.set(id, await infer(prefix, await input(sample(id).input), featureShapes) as unknown as DFineHeadFeatures);
  return cached.get(id)!;
}
function frozenUnchanged(value: DFineWeights) {
  return DFINE_FROZEN_NAMES.every(name => value[name].data.every((n, i) => Object.is(n, initial[name].data[i])));
}

export async function initialize(baseUrl: string, planDigest: string, dataDigest: string) {
  root = new URL(baseUrl); planSha256 = planDigest; datasetManifestSha256 = dataDigest;
  const plan = JSON.parse(new TextDecoder().decode(await bytes('plan.json', { sha256: planDigest })));
  validateImpactPlan(plan, dataDigest);
  const data = JSON.parse(new TextDecoder().decode(await bytes('data.json', { sha256: dataDigest }))) as { images: Sample[] };
  samples = data.images;
  if (samples.length !== 10 || new Set(samples.map(item => item.id)).size !== 10 ||
      ['train', 'tuning', 'evaluation'].some((split, i) => samples.filter(item => item.split === split).length !== [4, 2, 4][i])) {
    throw new Error('Invalid experiment split');
  }
  manifest = validateDetectorManifest(JSON.parse(new TextDecoder().decode(await bytes('model/model.json'))));
  const modelSpec = manifest.artifacts['detector.onnx'], prefixSpec = manifest.artifacts['frozen-prefix.onnx'];
  model = await bytes(`model/${modelSpec.file}`, modelSpec);
  const prefixBytes = await bytes(`model/${prefixSpec.file}`, prefixSpec);
  const headSpec = manifest.artifacts.headWeights;
  const saved = JSON.parse(new TextDecoder().decode(await bytes(`model/${headSpec.file}`, headSpec))) as {
    schemaVersion: number; config: Omit<DFineHeadConfig, 'projection'> & { projection: number[] }; weights: Record<string, Tensor>;
  };
  if (saved.schemaVersion !== 1) throw new Error('Unexpected head schema');
  config = { ...saved.config, projection: new Float32Array(saved.config.projection) };
  weights = Object.fromEntries(Object.entries(saved.weights).map(([name, value]) => [name, { shape: [...value.shape], data: new Float32Array(value.data) }]));
  initial = Object.fromEntries(Object.entries(weights).map(([name, value]) => [name, { shape: [...value.shape], data: new Float32Array(value.data) }]));
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL('runtime/', root).href;
  prefix = await ort.InferenceSession.create(prefixBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  return {
    schemaVersion: 1, planSha256, datasetManifestSha256,
    prefixSha256: await digest(prefixBytes), sourceModelSha256: await digest(model),
    weights: serialWeights(weights), config: { ...config, projection: Array.from(config.projection) },
    environment: { userAgent: navigator.userAgent, crossOriginIsolated, backend: 'wasm', threads: 1 },
  };
}

export async function measure(id: string, proposeUpdate = false) {
  const item = sample(id), ownFeatures = await features(id);
  const targets = { labels: new Int32Array(item.targets.labels), boxes: new Float32Array(item.targets.boxes) };
  const measured = await computeDFineHeadGradients(ownFeatures, targets, weights, config);
  const featureTensors = Object.fromEntries(Object.entries(featureShapes).map(([name, shape]) =>
    [name, { shape, data: Array.from(ownFeatures[name as keyof DFineHeadFeatures]) }]));
  const result = {
    schemaVersion: 1, planSha256, datasetManifestSha256, id, step, inputSha256: item.input.sha256,
    features: featureTensors, weights: serialWeights(weights), config: { ...config, projection: Array.from(config.projection) },
    prediction: serialPrediction(measured.prediction), losses: measured.losses,
    matches: { predictionIndices: Array.from(measured.matches.predictionIndices), targetIndices: Array.from(measured.matches.targetIndices) },
    gradients: serialWeights(measured.gradients), nextWeights: undefined as Record<string, Tensor> | undefined,
    frozenUnchanged: frozenUnchanged(weights),
  };
  if (proposeUpdate) {
    const trained = await trainDFineHead(ownFeatures, targets, weights, config, { steps: 1, learningRate: 1e-4 });
    if (trained.stepsCompleted !== 1 || !frozenUnchanged(trained.weights)) throw new Error('Unexpected update or frozen change');
    for (const name of DFINE_TRAINABLE_NAMES) for (let i = 0; i < weights[name].data.length; i++) {
      const expected = Math.fround(weights[name].data[i] - Math.fround(1e-4 * measured.gradients[name].data[i]));
      if (!Object.is(trained.weights[name].data[i], expected)) throw new Error('Recorded gradient differs from actual SGD update');
    }
    result.nextWeights = serialWeights(trained.weights);
    pending = { id, weights: trained.weights };
  }
  if (!result.frozenUnchanged) throw new Error('Frozen LQE changed');
  return result;
}

export function acceptUpdate(id: string, expectedStep: number) {
  const order = samples.filter(item => item.split === 'train').map(item => item.id);
  if (expectedStep !== step || step >= 12 || id !== order[step % 4] || pending?.id !== id) throw new Error('Unexpected browser update order');
  weights = pending.weights; pending = undefined; step++;
  return step;
}
export function checkpoint() { return serialWeights(weights); }

export async function exportModel() {
  if (step !== 12 || !frozenUnchanged(weights)) throw new Error('Incomplete browser training');
  const updated = Object.fromEntries(DFINE_TRAINABLE_NAMES.map(name => [name, weights[name]]));
  const exported = exportDFineOnnx(model, manifest.exportRecipe, updated);
  const blob = new Blob([new Uint8Array(exported).buffer], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob), anchor = document.createElement('a');
  anchor.href = url; anchor.download = 'model.onnx'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return { sha256: await digest(exported), bytes: exported.length };
}

// Called only in a separate browser context, after the training context is closed.
export async function verifyExport(baseUrl: string, artifact: Artifact, inputs: { id: string; input: Artifact }[]) {
  root = new URL(baseUrl);
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL('runtime/', root).href;
  const exported = await bytes(artifact.file, artifact);
  const session = await ort.InferenceSession.create(exported, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const images = [];
  try {
    for (const item of inputs) {
      const actual = await infer(session, await input(item.input), { logits: [300, 80], boxes: [300, 4] });
      images.push({ id: item.id, inputSha256: item.input.sha256, prediction: {
        logits: { shape: [300, 80], data: Array.from(actual.logits) }, boxes: { shape: [300, 4], data: Array.from(actual.boxes) },
      } });
    }
    return { schemaVersion: 1, modelSha256: await digest(exported), images };
  } finally { await session.release(); }
}
