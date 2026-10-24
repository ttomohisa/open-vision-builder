import { Array as JaxArray } from '@jax-js/jax';
import { describe, expect, it, vi } from 'vitest';
import { predictHead, trainHead, type TrainHeadOptions } from '../src/ml/head';
import type { HeadWeights } from '../src/ml/types';

const options: TrainHeadOptions = {
  classes: 2, featureSize: 2, epochs: 1, learningRate: 0.1, backend: 'wasm',
};
const features = () => new Float32Array([1, 2, 3, 4]);
const labels = () => new Int32Array([0, 1]);

describe('trainHead', () => {
  it('applies the independently derived first softmax gradient to the head', async () => {
    // At zero logits, P=[.5,.5]. X^T(P-Y)/2 is [[.5,-.5],[.5,-.5]].
    const result = await trainHead(
      new Float32Array([1, 2, 3, 4]),
      new Int32Array([0, 1]),
      { classes: 2, featureSize: 2, epochs: 1, learningRate: 0.1, backend: 'wasm' },
    );

    expect(result.head.weights).toEqual(new Float32Array([-0.05, 0.05, -0.05, 0.05]));
    expect(result.head.bias).toEqual(new Float32Array([0, 0]));
    expect(result.losses[0]).toBeCloseTo(Math.log(2), 6);
    expect(result.epochsCompleted).toBe(1);
    expect(result.cancelled).toBe(false);
    expect(result.backend).toBe('wasm');
  });

  it('reduces loss and separates a small learnable dataset', async () => {
    const input = new Float32Array([-2, 1, -1, 1, 1, 1, 2, 1]);
    const result = await trainHead(input, new Int32Array([0, 0, 1, 1]), { ...options, epochs: 12 });
    expect(result.losses).toHaveLength(12);
    expect(result.losses[0]).toBeCloseTo(Math.log(2), 6);
    expect(result.losses.at(-1)).toBeLessThan(0.4);
    expect(result.losses.every(Number.isFinite)).toBe(true);
    const logits = predictHead(input, result.head);
    expect(logits[0]).toBeGreaterThan(logits[1]);
    expect(logits[2]).toBeGreaterThan(logits[3]);
    expect(logits[4]).toBeLessThan(logits[5]);
    expect(logits[6]).toBeLessThan(logits[7]);
  });

  const invalid: {
    name: string;
    input?: Float32Array;
    targets?: Int32Array;
    options?: Partial<TrainHeadOptions>;
  }[] = [
    { name: 'one class', options: { classes: 1 } },
    { name: 'too many classes', options: { classes: 33 } },
    { name: 'fractional classes', options: { classes: 2.5 } },
    { name: 'zero feature size', options: { featureSize: 0 } },
    { name: 'too many features', options: { featureSize: 8193 } },
    { name: 'fractional feature size', options: { featureSize: 1.5 } },
    { name: 'zero epochs', options: { epochs: 0 } },
    { name: 'too many epochs', options: { epochs: 501 } },
    { name: 'fractional epochs', options: { epochs: 1.5 } },
    { name: 'zero learning rate', options: { learningRate: 0 } },
    { name: 'too large learning rate', options: { learningRate: 1.1 } },
    { name: 'NaN learning rate', options: { learningRate: NaN } },
    { name: 'empty data', input: new Float32Array(), targets: new Int32Array() },
    { name: 'wrong feature shape', input: new Float32Array([1, 2, 3]) },
    { name: 'negative class index', targets: new Int32Array([-1, 1]) },
    { name: 'out of range class index', targets: new Int32Array([0, 2]) },
    { name: 'missing training class', targets: new Int32Array([0, 0]) },
    { name: 'NaN feature', input: new Float32Array([1, NaN, 3, 4]) },
    { name: 'infinite feature', input: new Float32Array([1, 2, Infinity, 4]) },
    {
      name: 'wrong initial head shape',
      options: { initial: { weights: new Float32Array(3), bias: new Float32Array(2), featureSize: 2, classes: 2 } },
    },
    {
      name: 'non-finite initial head',
      options: { initial: { weights: new Float32Array([0, NaN, 0, 0]), bias: new Float32Array(2), featureSize: 2, classes: 2 } },
    },
    {
      name: 'incompatible initial classes',
      options: { initial: { weights: new Float32Array(6), bias: new Float32Array(3), featureSize: 2, classes: 3 } },
    },
  ];

  it.each(invalid)('rejects $name before reporting progress', async (invalidCase) => {
    const epochs: number[] = [];
    await expect(trainHead(
      invalidCase.input ?? features(),
      invalidCase.targets ?? labels(),
      { ...options, ...invalidCase.options, onEpoch: (epoch) => { epochs.push(epoch); } },
    )).rejects.toThrow();
    expect(epochs).toEqual([]);
  });

  it('returns an independent initial snapshot if cancelled before the first epoch', async () => {
    const initial: HeadWeights = {
      weights: new Float32Array([0.1, -0.1, 0.2, -0.2]),
      bias: new Float32Array([0.3, -0.3]), featureSize: 2, classes: 2,
    };
    const result = await trainHead(features(), labels(), { ...options, initial, isCancelled: () => true });
    expect(result.cancelled).toBe(true);
    expect(result.epochsCompleted).toBe(0);
    expect(result.losses).toEqual([]);
    expect(result.head.weights).toEqual(initial.weights);
    expect(result.head.bias).toEqual(initial.bias);
    expect(result.head.weights.buffer).not.toBe(initial.weights.buffer);
    expect(result.head.bias.buffer).not.toBe(initial.bias.buffer);
  });

  it('yields for cancellation and returns exactly the last completed update', async () => {
    let cancelled = false;
    const progress: number[] = [];
    const result = await trainHead(features(), labels(), {
      ...options,
      epochs: 20,
      isCancelled: () => cancelled,
      onEpoch: (epoch) => {
        progress.push(epoch);
        if (epoch === 1) setTimeout(() => { cancelled = true; }, 0);
      },
    });
    expect(result.cancelled).toBe(true);
    expect(result.epochsCompleted).toBe(1);
    expect(result.losses).toHaveLength(1);
    expect(progress).toEqual([1]);
    expect(result.head.weights).toEqual(new Float32Array([-0.05, 0.05, -0.05, 0.05]));
  });

  it('continues an initial head without mutating caller buffers or leaking state between calls', async () => {
    const input = features();
    const targets = labels();
    const first = await trainHead(input, targets, options);
    const initialCopy = new Float32Array(first.head.weights);
    const resumed = await trainHead(input, targets, { ...options, initial: first.head });
    const uninterrupted = await trainHead(input, targets, { ...options, epochs: 2 });
    expect(resumed.head.weights).not.toEqual(first.head.weights);
    expect(resumed.head.weights).toEqual(uninterrupted.head.weights);
    expect(resumed.head.bias).toEqual(uninterrupted.head.bias);
    expect(first.head.weights).toEqual(initialCopy);
    expect(input).toEqual(features());
    expect(targets).toEqual(labels());
    for (let i = 0; i < 3; i++) {
      const repeated = await trainHead(input, targets, options);
      expect(repeated.head.weights).toEqual(initialCopy);
      expect(repeated.head.weights.buffer).not.toBe(first.head.weights.buffer);
    }
  });

  it('releases retained jax arrays after completion, cancellation and callback errors', async () => {
    const seen = new Set<JaxArray>();
    const getRef = Object.getOwnPropertyDescriptor(JaxArray.prototype, 'ref')!.get!;
    const dispose = JaxArray.prototype.dispose;
    // Observe real reference counts without replacing either operation's behavior.
    const refSpy = vi.spyOn(JaxArray.prototype, 'ref', 'get').mockImplementation(function (this: JaxArray) {
      seen.add(this);
      return getRef.call(this) as JaxArray;
    });
    const disposeSpy = vi.spyOn(JaxArray.prototype, 'dispose').mockImplementation(function (this: JaxArray) {
      seen.add(this);
      dispose.call(this);
    });
    try {
      await trainHead(features(), labels(), options);
      await trainHead(features(), labels(), { ...options, isCancelled: () => true });
      await expect(trainHead(features(), labels(), {
        ...options, epochs: 2, onEpoch: async () => { throw new Error('progress consumer failed'); },
      })).rejects.toThrow('progress consumer failed');
    } finally {
      refSpy.mockRestore();
      disposeSpy.mockRestore();
    }
    expect(seen.size).toBeGreaterThan(0);
    expect([...seen].filter((array) => array.refCount !== 0)).toEqual([]);
  });
});

describe('predictHead', () => {
  it('computes independent feature-major matrix logits with bias', () => {
    const head: HeadWeights = {
      weights: new Float32Array([1, 2, 3, 4]), bias: new Float32Array([0.5, -0.5]),
      featureSize: 2, classes: 2,
    };
    expect(predictHead(new Float32Array([1, 2, -1, 1]), head)).toEqual(new Float32Array([7.5, 9.5, 2.5, 1.5]));
  });

  it('rejects malformed and non-finite prediction data', () => {
    const head: HeadWeights = { weights: new Float32Array(4), bias: new Float32Array(2), featureSize: 2, classes: 2 };
    expect(() => predictHead(new Float32Array([1]), head)).toThrow();
    expect(() => predictHead(new Float32Array([NaN, 1]), head)).toThrow();
    expect(() => predictHead(new Float32Array([1, 2]), { ...head, bias: new Float32Array([Infinity, 0]) })).toThrow();
  });
});
