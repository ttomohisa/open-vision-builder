import { Array as JaxArray } from '@jax-js/jax';
import { describe, expect, it, vi } from 'vitest';
import fixture from './fixtures/detection-head-reference.json';
import {
  computeDFineHeadGradients, predictDFineHead, trainDFineHead,
  DFINE_TRAINABLE_NAMES, type DFineWeights,
} from '../src/ml/detection/head';

const config = { ...fixture.config, projection: new Float32Array(fixture.config.projection) };
const features = {
  bboxInput: new Float32Array(fixture.features.bboxInput.data),
  classInput: new Float32Array(fixture.features.classInput.data),
  residualCorners: new Float32Array(fixture.features.residualCorners.data),
  references: new Float32Array(fixture.features.references.data),
};
const weights = (): DFineWeights => Object.fromEntries(Object.entries(fixture.weights)
  .map(([name, tensor]) => [name, { shape: tensor.shape, data: new Float32Array(tensor.data) }]));
const targets = (name: keyof typeof fixture.cases) => ({
  labels: new Int32Array(fixture.cases[name].targets.labels), boxes: new Float32Array(fixture.cases[name].targets.boxes),
});

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, absolute = 1e-5, relative = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  let failed = 0; let maxError = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    maxError = Math.max(maxError, error);
    if (!Number.isFinite(actual[i]) || error > absolute + relative * Math.abs(expected[i])) failed++;
  }
  expect({ failed, maxError: failed ? maxError : 0 }).toEqual({ failed: 0, maxError: 0 });
}

describe('frozen-prefix D-FINE final detector head', () => {
  it('matches an independent PyTorch distribution decoder and LQE forward', async () => {
    const result = await predictDFineHead(features, weights(), config);
    close(result.logits, fixture.cases.multiple.prediction.logits.data);
    close(result.boxes, fixture.cases.multiple.prediction.boxes.data);
    close(result.corners, fixture.cases.multiple.prediction.corners.data);
  });

  it.each(['empty', 'single', 'multiple'] as const)('matches PyTorch losses and all parameter gradients for %s targets', async (name) => {
    const result = await computeDFineHeadGradients(features, targets(name), weights(), config);
    const reference = fixture.cases[name];
    expect(Array.from(result.matches.predictionIndices)).toEqual(reference.matches.predictionIndices);
    expect(Array.from(result.matches.targetIndices)).toEqual(reference.matches.targetIndices);
    close([result.losses.vfl, result.losses.l1, result.losses.giou, result.losses.total],
      [reference.losses.vfl, reference.losses.l1, reference.losses.giou, reference.losses.total]);
    expect(Object.keys(result.gradients).sort()).toEqual([...DFINE_TRAINABLE_NAMES].sort());
    for (const key of DFINE_TRAINABLE_NAMES) close(result.gradients[key].data, reference.gradients[key].data);
  });

  it('uses the native zero ReLU derivative for exactly zero localization activations', async () => {
    const initial = weights();
    const firstLayer = DFINE_TRAINABLE_NAMES.slice(0, 2);
    for (const key of firstLayer) initial[key].data.fill(0);
    const result = await computeDFineHeadGradients(features, targets('single'), initial, config);
    // The second layer still receives a nonzero localization signal. PyTorch
    // ReLU blocks it at the exactly zero first-layer preactivation.
    expect(result.gradients['model.decoder.bbox_embed.2.layers.1.bias'].data.some(value => Math.abs(value) > 1e-8)).toBe(true);
    for (const key of firstLayer) expect(result.gradients[key].data).toEqual(new Float32Array(initial[key].data.length));
  });

  it('applies two independent SGD updates to localization and classifier while preserving frozen weights', async () => {
    const initial = weights();
    const copy = weights();
    const result = await trainDFineHead(features, targets('multiple'), initial, config, { steps: 2, learningRate: 1e-4 });
    expect(result.stepsCompleted).toBe(2);
    expect(result.cancelled).toBe(false);
    expect(result.losses).toHaveLength(2);
    for (let step = 0; step < 2; step++) close([result.losses[step]], [fixture.cases.multiple.steps[step].before.total]);
    for (const key of DFINE_TRAINABLE_NAMES) close(result.weights[key].data, fixture.cases.multiple.steps[1].weights[key].data);
    for (const key of Object.keys(initial)) {
      expect(initial[key].data).toEqual(copy[key].data);
      expect(result.weights[key].data.buffer).not.toBe(initial[key].data.buffer);
      if (!DFINE_TRAINABLE_NAMES.includes(key as typeof DFINE_TRAINABLE_NAMES[number])) expect(result.weights[key].data).toEqual(initial[key].data);
    }
    expect(result.metrics.localizationGradientL2).toBeGreaterThan(0);
    expect(result.metrics.localizationMaxChange).toBeGreaterThan(0);
    expect(result.metrics.localizationDeltaL2).toBeGreaterThan(0);
    expect(result.metrics.classificationDeltaL2).toBeGreaterThan(0);
    const after = await computeDFineHeadGradients(features, targets('multiple'), result.weights, config);
    expect(after.losses.total).toBeLessThan(result.losses[0]);
  });

  it('returns the last completed update on cancellation without mutating the caller', async () => {
    let cancelled = false;
    const result = await trainDFineHead(features, targets('multiple'), weights(), config, {
      steps: 4, learningRate: 1e-4, isCancelled: () => cancelled,
      onStep: () => { cancelled = true; },
    });
    expect(result.cancelled).toBe(true);
    expect(result.stepsCompleted).toBe(1);
    for (const key of DFINE_TRAINABLE_NAMES) close(result.weights[key].data, fixture.cases.multiple.steps[0].weights[key].data);
  });

  it('rejects mismatched model tensors and feature shapes before training', async () => {
    const malformed = weights();
    malformed['model.decoder.bbox_embed.2.layers.2.weight'].shape = [3, 4];
    await expect(trainDFineHead(features, targets('single'), malformed, config, { steps: 1, learningRate: 1e-4 })).rejects.toThrow();
    await expect(predictDFineHead({ ...features, references: new Float32Array([NaN]) }, weights(), config)).rejects.toThrow();
    await expect(trainDFineHead(features, targets('single'), weights(), config, { steps: 0, learningRate: 1e-4 })).rejects.toThrow();
  });

  it('uses a stable snapshot when the caller changes inputs, weights and options during progress', async () => {
    const input = Object.fromEntries(Object.entries(features).map(([key, value]) => [key, new Float32Array(value)])) as typeof features;
    const initial = weights();
    const modelConfig = { ...config, projection: new Float32Array(config.projection) };
    const settings = {
      steps: 2, learningRate: 1e-4,
      onStep: () => {
        settings.learningRate = 0.1; settings.steps = 1;
        modelConfig.classes = 1;
        input.bboxInput.fill(100);
        initial[DFINE_TRAINABLE_NAMES[0]].data.fill(100);
      },
    };
    const result = await trainDFineHead(input, targets('multiple'), initial, modelConfig, settings);
    expect(result.stepsCompleted).toBe(2);
    for (const key of DFINE_TRAINABLE_NAMES) close(result.weights[key].data, fixture.cases.multiple.steps[1].weights[key].data);
  });

  it('releases retained arrays on completion, cancellation and progress callback errors', async () => {
    const seen = new Set<JaxArray>();
    const getRef = Object.getOwnPropertyDescriptor(JaxArray.prototype, 'ref')!.get!;
    const dispose = JaxArray.prototype.dispose;
    const refs = vi.spyOn(JaxArray.prototype, 'ref', 'get').mockImplementation(function (this: JaxArray) {
      seen.add(this); return getRef.call(this) as JaxArray;
    });
    const releases = vi.spyOn(JaxArray.prototype, 'dispose').mockImplementation(function (this: JaxArray) {
      seen.add(this); dispose.call(this);
    });
    try {
      await trainDFineHead(features, targets('multiple'), weights(), config, { steps: 1, learningRate: 1e-4 });
      await trainDFineHead(features, targets('empty'), weights(), config, { steps: 1, learningRate: 1e-4, isCancelled: () => true });
      await expect(trainDFineHead(features, targets('single'), weights(), config, {
        steps: 1, learningRate: 1e-4, onStep: () => { throw new Error('consumer failure'); },
      })).rejects.toThrow('consumer failure');
    } finally {
      refs.mockRestore(); releases.mockRestore();
    }
    expect(seen.size).toBeGreaterThan(0);
    expect([...seen].filter((array) => array.refCount !== 0)).toEqual([]);
  });
});
