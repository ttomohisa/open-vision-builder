import {
  blockUntilReady, init, jit, nn, numpy as np, tree, valueAndGrad, type OwnedFunction,
} from '@jax-js/jax';
import { applyUpdates, sgd, type OptState } from '@jax-js/optax';
import type { HeadWeights } from './types';

export interface TrainHeadOptions {
  classes: number;
  featureSize: number;
  epochs: number;
  learningRate: number;
  backend?: 'wasm' | 'webgpu';
  initial?: HeadWeights;
  isCancelled?: () => boolean;
  onEpoch?: (epoch: number, loss: number) => void | Promise<void>;
}

export interface TrainHeadResult {
  head: HeadWeights;
  losses: number[];
  epochsCompleted: number;
  cancelled: boolean;
  backend: string;
}

type Params = { weights: np.Array; bias: np.Array };
type Step = OwnedFunction<(
  params: Params, state: OptState, features: np.Array, targets: np.Array,
) => [np.Array, Params, OptState]>;

function integerInRange(value: number, min: number, max: number, name: string): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer from ${min} to ${max}.`);
  }
}

function finiteArray(values: Float32Array, name: string): void {
  if (!(values instanceof Float32Array)) throw new Error(`${name} must be Float32Array.`);
  for (const value of values) {
    if (!Number.isFinite(value)) throw new Error(`${name} contains a non-finite value.`);
  }
}

function validateHead(head: HeadWeights): void {
  integerInRange(head.classes, 2, 32, 'Class count');
  integerInRange(head.featureSize, 1, 8192, 'Feature size');
  finiteArray(head.weights, 'Head weights');
  finiteArray(head.bias, 'Head bias');
  if (head.weights.length !== head.featureSize * head.classes || head.bias.length !== head.classes) {
    throw new Error('Head weights and bias do not match the declared shape.');
  }
}

// All arguments are consumed once. Only the first argument is differentiated.
function loss(params: Params, features: np.Array, targets: np.Array): np.Array {
  const logits = np.matmul(features, params.weights).add(params.bias);
  return nn.logSoftmax(logits, -1).mul(targets).sum(-1).neg().mean();
}

const lossAndGrad = valueAndGrad(loss);

/** Consume one owned reference and copy the readback before releasing its owner. */
async function readFloat32(array: np.Array): Promise<Float32Array> {
  let data: Awaited<ReturnType<np.Array['data']>>;
  try {
    data = await array.data(); // data() consumes the reference on success.
  } catch (error) {
    array.dispose();
    throw error;
  }
  if (!(data instanceof Float32Array)) throw new Error('Training produced a non-FP32 tensor.');
  return new Float32Array(data);
}

/**
 * Train a feature-major [featureSize, classes] linear head on a full batch.
 * Each loss is evaluated immediately before its corresponding SGD update.
 */
export async function trainHead(
  features: Float32Array,
  labels: Int32Array,
  options: TrainHeadOptions,
): Promise<TrainHeadResult> {
  const { classes, featureSize, epochs, learningRate, onEpoch, isCancelled } = options;
  const backend = options.backend ?? 'wasm';
  integerInRange(classes, 2, 32, 'Class count');
  integerInRange(featureSize, 1, 8192, 'Feature size');
  integerInRange(epochs, 1, 500, 'Epoch count');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
    throw new Error('Learning rate must be finite, greater than 0 and at most 1.');
  }
  if (backend !== 'wasm' && backend !== 'webgpu') throw new Error('Unsupported training backend.');
  finiteArray(features, 'Training features');
  if (!(labels instanceof Int32Array) || labels.length === 0) {
    throw new Error('Training requires a nonempty Int32Array of labels.');
  }
  if (features.length !== labels.length * featureSize) {
    throw new Error('Training features do not match the label count and feature size.');
  }
  const present = new Uint8Array(classes);
  const targets = new Float32Array(labels.length * classes);
  labels.forEach((label, index) => {
    if (label < 0 || label >= classes) throw new Error('A training label is outside the class range.');
    present[label] = 1;
    targets[index * classes + label] = 1;
  });
  if (present.some((count) => count === 0)) throw new Error('Every class needs at least one training example.');
  if (options.initial) {
    validateHead(options.initial);
    if (options.initial.classes !== classes || options.initial.featureSize !== featureSize) {
      throw new Error('The initial head does not match the requested class count and feature size.');
    }
  }
  // Snapshot caller buffers before the first asynchronous boundary.
  const input = new Float32Array(features);
  const initialWeights = options.initial ? new Float32Array(options.initial.weights) : new Float32Array(featureSize * classes);
  const initialBias = options.initial ? new Float32Array(options.initial.bias) : new Float32Array(classes);
  const devices = await init(backend);
  if (!devices.includes(backend)) throw new Error(`${backend} is unavailable for training.`);

  let x: np.Array | undefined;
  let y: np.Array | undefined;
  let params: Params | undefined;
  let state: OptState | undefined;
  let step: Step | undefined;
  const losses: number[] = [];
  let cancelled = false;
  try {
    x = np.array(input, { shape: [labels.length, featureSize], dtype: np.float32, device: backend });
    y = np.array(targets, { shape: [labels.length, classes], dtype: np.float32, device: backend });
    params = {
      weights: np.array(initialWeights, { shape: [featureSize, classes], dtype: np.float32, device: backend }),
      bias: np.array(initialBias, { shape: [classes], dtype: np.float32, device: backend }),
    };
    const solver = sgd(learningRate);
    state = solver.init(tree.ref(params));
    step = jit((p: Params, optState: OptState, batch: np.Array, oneHot: np.Array): [np.Array, Params, OptState] => {
      const [value, gradient] = lossAndGrad(tree.ref(p), batch, oneHot);
      const [updates, nextState] = solver.update(gradient, optState);
      return [value, applyUpdates(p, updates), nextState];
    }); // In 0.1.27, jit follows the explicitly placed inputs; it has no device option.

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (isCancelled?.()) {
        cancelled = true;
        break;
      }
      let value: np.Array | undefined;
      let nextParams: Params | undefined;
      let nextState: OptState | undefined;
      try {
        // Retain the previous state until the complete update has finished.
        [value, nextParams, nextState] = step(tree.ref(params), tree.ref(state), x.ref, y.ref);
        await blockUntilReady([nextParams, nextState]); // Does not consume references.
        const epochLoss = (await readFloat32(value.ref))[0];
        if (!Number.isFinite(epochLoss)) throw new Error('Training loss became non-finite.');
        tree.dispose(params);
        tree.dispose(state);
        params = nextParams;
        state = nextState;
        nextParams = undefined;
        nextState = undefined;
        losses.push(epochLoss);
        await onEpoch?.(epoch + 1, epochLoss);
      } finally {
        value?.dispose();
        tree.dispose(nextParams);
        tree.dispose(nextState);
      }
      // A macrotask boundary lets Worker messages set the cancellation flag.
      if (epoch + 1 < epochs) await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }

    const [weights, bias] = await Promise.all([
      readFloat32(params.weights.ref), readFloat32(params.bias.ref),
    ]);
    const head = { weights, bias, featureSize, classes };
    validateHead(head);
    return { head, losses, epochsCompleted: losses.length, cancelled, backend };
  } finally {
    step?.dispose();
    tree.dispose(params);
    tree.dispose(state);
    x?.dispose();
    y?.dispose();
  }
}

/** Independent JavaScript inference; returns row-major [batch, classes] logits. */
export function predictHead(features: Float32Array, head: HeadWeights): Float32Array {
  validateHead(head);
  finiteArray(features, 'Prediction features');
  if (features.length === 0 || features.length % head.featureSize !== 0) {
    throw new Error('Prediction features do not match the head feature size.');
  }
  const count = features.length / head.featureSize;
  const output = new Float32Array(count * head.classes);
  for (let row = 0; row < count; row++) {
    for (let cls = 0; cls < head.classes; cls++) {
      let sum = head.bias[cls];
      for (let feature = 0; feature < head.featureSize; feature++) {
        sum += features[row * head.featureSize + feature] * head.weights[feature * head.classes + cls];
      }
      output[row * head.classes + cls] = sum;
    }
  }
  finiteArray(output, 'Prediction logits');
  return output;
}
