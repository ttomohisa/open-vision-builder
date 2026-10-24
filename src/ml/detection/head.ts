import { init, jit, lax, nn, numpy as np, tree, valueAndGrad } from '@jax-js/jax';
import { buildDFineLossTargets, dFineLoss, makeDFineLossArrays, type DFineLossArrays } from './loss';
import { validateDetectionTargets, type DetectionTargets, type MatchedIndices } from './matching';

const PREFIX = 'model.decoder.';
export const DFINE_TRAINABLE_NAMES = [
  'model.decoder.bbox_embed.2.layers.0.weight', 'model.decoder.bbox_embed.2.layers.0.bias',
  'model.decoder.bbox_embed.2.layers.1.weight', 'model.decoder.bbox_embed.2.layers.1.bias',
  'model.decoder.bbox_embed.2.layers.2.weight', 'model.decoder.bbox_embed.2.layers.2.bias',
  'model.decoder.class_embed.2.weight', 'model.decoder.class_embed.2.bias',
] as const;
export const DFINE_FROZEN_NAMES = [
  'model.decoder.lqe_layers.2.reg_conf.layers.0.weight', 'model.decoder.lqe_layers.2.reg_conf.layers.0.bias',
  'model.decoder.lqe_layers.2.reg_conf.layers.1.weight', 'model.decoder.lqe_layers.2.reg_conf.layers.1.bias',
] as const;

/** Source-oriented dense weights are always [output, input], with no hidden transpose in the artifact. */
export interface DFineTensor { shape: number[]; data: Float32Array }
export type DFineWeights = Record<string, DFineTensor>;
export interface DFineHeadConfig {
  queries: number;
  hiddenSize: number;
  classes: number;
  bins: number;
  regScale: number;
  topK: number;
  projection: Float32Array;
}
/** Flat tensors for a single image. The complete frozen prefix supplies these four values. */
export interface DFineHeadFeatures {
  bboxInput: Float32Array;
  classInput: Float32Array;
  residualCorners: Float32Array;
  references: Float32Array;
}
export interface DFinePrediction { logits: Float32Array; boxes: Float32Array; corners: Float32Array }
export interface DFineLossValues { vfl: number; l1: number; giou: number; total: number }
export interface DFineGradientResult {
  prediction: DFinePrediction;
  losses: DFineLossValues;
  gradients: DFineWeights;
  matches: MatchedIndices;
}
export interface DFineTrainingMetrics {
  localizationGradientL2: number;
  classificationGradientL2: number;
  localizationMaxChange: number;
  classificationMaxChange: number;
  localizationDeltaL2: number;
  classificationDeltaL2: number;
}
export interface DFineTrainOptions {
  steps: number;
  learningRate: number;
  backend?: 'wasm';
  isCancelled?: () => boolean;
  onStep?: (step: number, loss: number) => void | Promise<void>;
}
export interface DFineTrainResult {
  weights: DFineWeights;
  losses: number[];
  stepsCompleted: number;
  cancelled: boolean;
  backend: 'wasm';
  metrics: DFineTrainingMetrics;
}

type Arrays = Record<string, np.Array>;
interface FeatureArrays { bboxInput: np.Array; classInput: np.Array; residualCorners: np.Array; references: np.Array }
interface FrozenArrays { weights: Arrays; projection: np.Array }
interface PredictionArrays { logits: np.Array; boxes: np.Array; corners: np.Array }
interface Snapshot { features: DFineHeadFeatures; weights: DFineWeights; config: DFineHeadConfig }

function integer(value: number, minimum: number, maximum: number, name: string): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`${name} is outside the supported range.`);
}
function finite(values: Float32Array, size: number, name: string): void {
  if (!(values instanceof Float32Array) || values.length !== size || !values.every(Number.isFinite)) {
    throw new Error(`${name} has an invalid shape or non-finite values.`);
  }
}
function cloneWeights(weights: DFineWeights): DFineWeights {
  return Object.fromEntries(Object.entries(weights).map(([name, tensor]) => [name, {
    shape: [...tensor.shape], data: new Float32Array(tensor.data),
  }]));
}
function snapshot(features: DFineHeadFeatures, weights: DFineWeights, config: DFineHeadConfig): Snapshot {
  integer(config.queries, 1, 300, 'Query count');
  integer(config.hiddenSize, 1, 1024, 'Hidden size');
  integer(config.classes, 1, 1000, 'Class count');
  integer(config.bins, 2, 65, 'Distribution bins');
  integer(config.topK, 1, config.bins, 'LQE top K');
  if (!Number.isFinite(config.regScale) || config.regScale <= 0) throw new Error('Regression scale must be positive and finite.');
  finite(config.projection, config.bins, 'Distribution projection');
  const h = config.hiddenSize;
  const expected: Record<string, number[]> = {};
  for (let i = 0; i < 3; i++) {
    const output = i === 2 ? 4 * config.bins : h;
    expected[`${PREFIX}bbox_embed.2.layers.${i}.weight`] = [output, h];
    expected[`${PREFIX}bbox_embed.2.layers.${i}.bias`] = [output];
  }
  expected[`${PREFIX}class_embed.2.weight`] = [config.classes, h];
  expected[`${PREFIX}class_embed.2.bias`] = [config.classes];
  const lqeHidden = weights[DFINE_FROZEN_NAMES[0]]?.shape[0];
  integer(lqeHidden, 1, 1024, 'LQE hidden size');
  expected[DFINE_FROZEN_NAMES[0]] = [lqeHidden, 4 * (config.topK + 1)];
  expected[DFINE_FROZEN_NAMES[1]] = [lqeHidden];
  expected[DFINE_FROZEN_NAMES[2]] = [1, lqeHidden];
  expected[DFINE_FROZEN_NAMES[3]] = [1];
  for (const [name, shape] of Object.entries(expected)) {
    const tensor = weights[name];
    if (!tensor || tensor.shape.length !== shape.length || shape.some((value, i) => tensor.shape[i] !== value)) {
      throw new Error(`Pretrained tensor ${name} has an incompatible shape.`);
    }
    finite(tensor.data, shape.reduce((a, b) => a * b, 1), name);
  }
  finite(features.bboxInput, config.queries * h, 'Bounding-box input');
  finite(features.classInput, config.queries * h, 'Classification input');
  finite(features.residualCorners, config.queries * config.bins * 4, 'Residual distribution');
  finite(features.references, config.queries * 4, 'Reference boxes');
  for (let i = 0; i < config.queries; i++) {
    if (features.references[4 * i + 2] <= 0 || features.references[4 * i + 3] <= 0) throw new Error('Reference boxes must have positive size.');
  }
  return {
    features: {
      bboxInput: new Float32Array(features.bboxInput), classInput: new Float32Array(features.classInput),
      residualCorners: new Float32Array(features.residualCorners), references: new Float32Array(features.references),
    },
    weights: cloneWeights(weights),
    config: { ...config, projection: new Float32Array(config.projection) },
  };
}

function allocate(data: Snapshot): { params: Arrays; fixed: FrozenArrays; features: FeatureArrays } {
  const { config, weights, features } = data;
  const array = (values: Float32Array, shape: number[]) => np.array(new Float32Array(values), { shape, dtype: np.float32, device: 'wasm' });
  return {
    params: Object.fromEntries(DFINE_TRAINABLE_NAMES.map((name) => [name, array(weights[name].data, weights[name].shape)])),
    fixed: {
      weights: Object.fromEntries(DFINE_FROZEN_NAMES.map((name) => [name, array(weights[name].data, weights[name].shape)])),
      projection: array(config.projection, [config.bins]),
    },
    features: {
      bboxInput: array(features.bboxInput, [config.queries, config.hiddenSize]),
      classInput: array(features.classInput, [config.queries, config.hiddenSize]),
      residualCorners: array(features.residualCorners, [config.queries, 4 * config.bins]),
      references: array(features.references, [config.queries, 4]),
    },
  };
}

function dense(input: np.Array, weights: Arrays, name: string): np.Array {
  return np.matmul(input, weights[`${name}.weight`].transpose()).add(weights[`${name}.bias`]);
}

function relu(input: np.Array): np.Array {
  // PyTorch ReLU has zero derivative at zero; jax maximum splits that tie.
  return np.where(np.greater(input.ref, 0), input, 0);
}

/**
 * This is the unchanged final pretrained inference head, including cumulative
 * distribution refinement and frozen LQE. Earlier decoder tensors are fixed
 * inputs: final-head tuning cannot backpropagate through or update the prefix.
 */
function forward(params: Arrays, fixed: FrozenArrays, features: FeatureArrays, config: DFineHeadConfig): PredictionArrays {
  let output = features.bboxInput;
  for (let i = 0; i < 3; i++) {
    output = dense(output, params, `${PREFIX}bbox_embed.2.layers.${i}`);
    if (i < 2) output = relu(output);
  }
  const corners = output.add(features.residualCorners);
  const probabilities = nn.softmax(corners.ref.reshape([config.queries, 4, config.bins]), -1);
  const distances = probabilities.ref.mul(fixed.projection).sum(-1);
  const centers = features.references.ref.slice([], [0, 2]);
  const scaledSizes = features.references.slice([], [2, 4]).div(config.regScale);
  const topLeft = centers.ref.sub(distances.ref.slice([], [0, 2]).add(config.regScale / 2).mul(scaledSizes.ref));
  const bottomRight = centers.add(distances.slice([], [2, 4]).add(config.regScale / 2).mul(scaledSizes));
  const boxes = np.concatenate([topLeft.ref.add(bottomRight.ref).div(2), bottomRight.sub(topLeft)], -1);
  const [top, indices] = lax.topK(probabilities, config.topK, -1);
  indices.dispose();
  const statistics = np.concatenate([top.ref, top.mean(-1, { keepdims: true })], -1)
    .reshape([config.queries, 4 * (config.topK + 1)]);
  const quality = dense(relu(dense(statistics, fixed.weights, `${PREFIX}lqe_layers.2.reg_conf.layers.0`)),
    fixed.weights, `${PREFIX}lqe_layers.2.reg_conf.layers.1`);
  const logits = dense(features.classInput, params, `${PREFIX}class_embed.2`).add(quality);
  return { logits, boxes, corners };
}

async function read(array: np.Array): Promise<Float32Array> {
  const owned = array.ref;
  let values: Awaited<ReturnType<np.Array['data']>>;
  try { values = await owned.data(); }
  catch (error) { owned.dispose(); throw error; }
  if (!(values instanceof Float32Array) || !values.every(Number.isFinite)) throw new Error('D-FINE produced a non-finite or non-FP32 tensor.');
  return new Float32Array(values);
}
async function readPrediction(output: PredictionArrays): Promise<DFinePrediction> {
  try {
    const [logits, boxes, corners] = await Promise.all([read(output.logits), read(output.boxes), read(output.corners)]);
    return { logits, boxes, corners };
  } finally { tree.dispose(output); }
}
async function backend(options: { backend?: 'wasm' }): Promise<void> {
  if (options.backend !== undefined && options.backend !== 'wasm') throw new Error('Only the verified WASM detection backend is supported.');
  if (!(await init('wasm')).includes('wasm')) throw new Error('WASM is unavailable for detection training.');
}

export async function predictDFineHead(
  features: DFineHeadFeatures, weights: DFineWeights, config: DFineHeadConfig, options: { backend?: 'wasm' } = {},
): Promise<DFinePrediction> {
  const data = snapshot(features, weights, config);
  await backend(options);
  const arrays = allocate(data);
  try { return await readPrediction(forward(tree.ref(arrays.params), tree.ref(arrays.fixed), tree.ref(arrays.features), data.config)); }
  finally { tree.dispose(arrays); }
}

/** Returns actual autodiff parameter gradients with assignment and detached targets evaluated outside autodiff. */
export async function computeDFineHeadGradients(
  features: DFineHeadFeatures, targets: DetectionTargets, weights: DFineWeights, config: DFineHeadConfig,
  options: { backend?: 'wasm' } = {},
): Promise<DFineGradientResult> {
  const data = snapshot(features, weights, config);
  validateDetectionTargets(targets, config.classes);
  const copiedTargets = { labels: new Int32Array(targets.labels), boxes: new Float32Array(targets.boxes) };
  await backend(options);
  const arrays = allocate(data);
  let lossArrays: DFineLossArrays | undefined;
  const differentiate = jit(valueAndGrad((params: Arrays, fixed: FrozenArrays, input: FeatureArrays, labels: DFineLossArrays) => {
    const result = forward(params, fixed, input, data.config);
    result.corners.dispose();
    return dFineLoss(result.logits, result.boxes, labels);
  }, { hasAux: true }));
  let value: np.Array | undefined; let components: np.Array | undefined; let gradients: Arrays | undefined;
  try {
    const prediction = await readPrediction(forward(tree.ref(arrays.params), tree.ref(arrays.fixed), tree.ref(arrays.features), data.config));
    const lossTargets = buildDFineLossTargets(prediction, copiedTargets, data.config);
    lossArrays = makeDFineLossArrays(lossTargets, data.config, 'wasm');
    [[value, components], gradients] = differentiate(tree.ref(arrays.params), tree.ref(arrays.fixed), tree.ref(arrays.features), tree.ref(lossArrays));
    const [total, terms, gradientValues] = await Promise.all([
      read(value), read(components), Promise.all(DFINE_TRAINABLE_NAMES.map((name) => read(gradients![name]))),
    ]);
    return {
      prediction,
      losses: { total: total[0], vfl: terms[0], l1: terms[1], giou: terms[2] },
      gradients: Object.fromEntries(DFINE_TRAINABLE_NAMES.map((name, i) => [name, { shape: [...data.weights[name].shape], data: gradientValues[i] }])),
      matches: lossTargets.matches,
    };
  } finally {
    value?.dispose(); components?.dispose(); tree.dispose(gradients);
    differentiate.dispose(); tree.dispose(lossArrays); tree.dispose(arrays);
  }
}

/**
 * Plain SGD on eight existing final-layer tensors. This API requires all initial
 * pretrained tensors and supplies no random initialization or class remapping.
 * Frozen tensors, source names and source orientations are retained for ONNX export.
 */
export async function trainDFineHead(
  features: DFineHeadFeatures, targets: DetectionTargets, initialWeights: DFineWeights,
  config: DFineHeadConfig, options: DFineTrainOptions,
): Promise<DFineTrainResult> {
  const { steps, learningRate, isCancelled, onStep } = options;
  const runtime = { backend: options.backend };
  integer(steps, 1, 500, 'Training steps');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.1) throw new Error('Invalid detection learning rate.');
  const data = snapshot(features, initialWeights, config);
  validateDetectionTargets(targets, config.classes);
  const copiedTargets = { labels: new Int32Array(targets.labels), boxes: new Float32Array(targets.boxes) };
  const original = cloneWeights(data.weights);
  let weights = data.weights;
  const losses: number[] = [];
  const metrics: DFineTrainingMetrics = {
    localizationGradientL2: 0, classificationGradientL2: 0, localizationMaxChange: 0,
    classificationMaxChange: 0, localizationDeltaL2: 0, classificationDeltaL2: 0,
  };
  let cancelled = false;
  await backend(runtime);
  for (let step = 0; step < steps; step++) {
    if (isCancelled?.()) { cancelled = true; break; }
    const result = await computeDFineHeadGradients(data.features, copiedTargets, weights, data.config, runtime);
    if (isCancelled?.()) { cancelled = true; break; }
    const next = cloneWeights(weights);
    let localizationSquared = 0; let classificationSquared = 0;
    for (const name of DFINE_TRAINABLE_NAMES) {
      const gradient = result.gradients[name].data;
      const localized = name.includes('.bbox_embed.');
      for (let i = 0; i < gradient.length; i++) {
        // Match an FP32 multiply followed by an FP32 subtract; no optimizer state is hidden.
        const updated = Math.fround(weights[name].data[i] - Math.fround(learningRate * gradient[i]));
        if (!Number.isFinite(updated)) throw new Error('D-FINE SGD produced a non-finite parameter.');
        next[name].data[i] = updated;
        if (localized) localizationSquared += gradient[i] ** 2;
        else classificationSquared += gradient[i] ** 2;
      }
    }
    weights = next;
    metrics.localizationGradientL2 = Math.max(metrics.localizationGradientL2, Math.sqrt(localizationSquared));
    metrics.classificationGradientL2 = Math.max(metrics.classificationGradientL2, Math.sqrt(classificationSquared));
    losses.push(result.losses.total);
    await onStep?.(step + 1, result.losses.total);
    if (step + 1 < steps) await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  for (const name of DFINE_TRAINABLE_NAMES) {
    const key = name.includes('.bbox_embed.') ? 'localizationMaxChange' : 'classificationMaxChange';
    const normKey = name.includes('.bbox_embed.') ? 'localizationDeltaL2' : 'classificationDeltaL2';
    for (let i = 0; i < weights[name].data.length; i++) {
      const change = weights[name].data[i] - original[name].data[i];
      metrics[key] = Math.max(metrics[key], Math.abs(change));
      metrics[normKey] += change ** 2;
    }
  }
  metrics.localizationDeltaL2 = Math.sqrt(metrics.localizationDeltaL2);
  metrics.classificationDeltaL2 = Math.sqrt(metrics.classificationDeltaL2);
  return { weights, losses, stepsCompleted: losses.length, cancelled, backend: 'wasm', metrics };
}
