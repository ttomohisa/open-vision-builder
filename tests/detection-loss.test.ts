import { defaultDevice, grad, jit, numpy as np, tree, valueAndGrad } from '@jax-js/jax';
import { describe, expect, it } from 'vitest';
import { buildDFineLossTargets, dFineLoss, makeDFineLossArrays, type DFineLossArrays } from '../src/ml/detection/loss';
import tolerance from './fixtures/detector-training-tolerances.json';

defaultDevice('wasm');

describe('final-output D-FINE loss', () => {
  it.each([
    { logit: 100, expectedLoss: 1_799_925, positiveGradient: 0, negativeGradient: 0.75 },
    { logit: -100, expectedLoss: 100, positiveGradient: -1, negativeGradient: 0 },
    { logit: 0, expectedLoss: 4500.8125 * Math.log(2), positiveGradient: -0.5, negativeGradient: 0.09375 },
  ])('keeps detector-sized JIT BCE finite with native derivatives at logit $logit', async ({
    logit, expectedLoss, positiveGradient, negativeGradient,
  }) => {
    // Small tensors can use jax's inline JS-double path and hide FP32 exp overflow.
    const shape = { queries: 300, classes: 80 };
    const boxes = new Float32Array(1200);
    for (let query = 0; query < 300; query++) boxes.set([0.5, 0.5, 0.25, 0.25], query * 4);
    boxes.set([0.5, 0.5, 0.5, 0.5]); // Query 0 is the unique exact match.
    const prediction = {
      logits: new Float32Array(24_000).fill(logit), boxes,
    };
    const prepared = buildDFineLossTargets(prediction, {
      labels: new Int32Array([0]), boxes: new Float32Array([0.5, 0.5, 0.5, 0.5]),
    }, shape);
    expect(Array.from(prepared.matches.predictionIndices)).toEqual([0]);
    const arrays = makeDFineLossArrays(prepared, shape, 'wasm');
    const inputs = {
      logits: np.array(prediction.logits, { shape: [300, 80], device: 'wasm' }),
      boxes: np.array(prediction.boxes, { shape: [300, 4], device: 'wasm' }),
    };
    const differentiate = jit(valueAndGrad((logits: np.Array, boxes: np.Array, labels: DFineLossArrays) =>
      dFineLoss(logits, boxes, labels), { hasAux: true }));
    let result: ReturnType<typeof differentiate> | undefined;
    try {
      result = differentiate(inputs.logits.ref, inputs.boxes.ref, tree.ref(arrays));
      const [[total, components], gradient] = result;
      const [loss, terms, values] = await Promise.all([total.ref.data(), components.ref.data(), gradient.ref.data()]);
      // One IoU=1 positive and 23,999 negatives; VFL weights are detached.
      expect(Number.isFinite(loss[0])).toBe(true);
      // The existing pretrained-loss gate includes FP32 reduction error over 24,000 values.
      expect(Math.abs(loss[0] - expectedLoss)).toBeLessThanOrEqual(
        tolerance.pretrainedLoss.absolute + tolerance.pretrainedLoss.relative * Math.abs(expectedLoss),
      );
      expect(Array.from(terms)).toEqual([loss[0], 0, 0]);
      expect(values.every(Number.isFinite)).toBe(true);
      expect(values[0]).toBe(positiveGradient);
      expect(Array.from(values.subarray(1))).toEqual(new Array(23_999).fill(negativeGradient));
    } finally {
      tree.dispose(result); differentiate.dispose(); tree.dispose(inputs); tree.dispose(arrays);
    }
  });

  it('has zero box gradient when a matched box is exactly correct', async () => {
    const prediction = { logits: new Float32Array([0]), boxes: new Float32Array([0.5, 0.5, 0.4, 0.4]) };
    const prepared = buildDFineLossTargets(prediction, {
      labels: new Int32Array([0]), boxes: new Float32Array(prediction.boxes),
    }, { queries: 1, classes: 1 });
    const arrays = makeDFineLossArrays(prepared, { queries: 1, classes: 1 }, 'wasm');
    const boxes = np.array(prediction.boxes, { shape: [1, 4], device: 'wasm' });
    const differentiate = jit(valueAndGrad((input: np.Array, labels: DFineLossArrays) => {
      const [total, components] = dFineLoss(np.zeros([1, 1], { device: 'wasm' }), input, labels);
      components.dispose();
      return total;
    }));
    let result: ReturnType<typeof differentiate> | undefined;
    try {
      result = differentiate(boxes.ref, tree.ref(arrays));
      const [total, gradient] = result;
      expect((await total.ref.data())[0]).toBeCloseTo(Math.log(2), 6);
      // PyTorch L1 chooses subgradient zero at zero, and identical-box GIoU is stationary.
      expect(Array.from(await gradient.ref.data())).toEqual([0, 0, 0, 0]);
    } finally {
      tree.dispose(result); differentiate.dispose(); boxes.dispose(); tree.dispose(arrays);
    }
  });

  it('uses detached VFL probability weights and IoU targets, with native box losses', async () => {
    const prediction = { logits: new Float32Array([0, 0]), boxes: new Float32Array([0.5, 0.5, 0.4, 0.4]) };
    const targets = { labels: new Int32Array([0]), boxes: new Float32Array([0.5, 0.5, 0.4, 0.4]) };
    const prepared = buildDFineLossTargets(prediction, targets, { queries: 1, classes: 2 });
    expect(prepared.targetScores).toEqual(new Float32Array([1, 0]));
    expect(prepared.vflWeights).toEqual(new Float32Array([1, 0.1875]));
    const inputs = {
      logits: np.array(prediction.logits, { shape: [1, 2] }),
      boxes: np.array(prediction.boxes, { shape: [1, 4] }),
    };
    const arrays = makeDFineLossArrays(prepared, { queries: 1, classes: 2 }, 'wasm');
    try {
      const [total, components] = dFineLoss(inputs.logits.ref, inputs.boxes.ref, tree.ref(arrays));
      expect((await total.data())[0]).toBeCloseTo(1.1875 * Math.log(2), 6);
      expect(Array.from(await components.data())).toEqual([expect.closeTo(1.1875 * Math.log(2), 6), 0, 0]);
      const gradient = grad((logits: np.Array) => {
        const [loss, rest] = dFineLoss(logits, inputs.boxes.ref, tree.ref(arrays));
        rest.dispose();
        return loss;
      })(inputs.logits.ref);
      // Differentiating the VFL weight would add a spurious term to the negative logit.
      expect(Array.from(await gradient.data())).toEqual([-0.5, 0.09375]);
    } finally {
      tree.dispose(inputs);
      tree.dispose(arrays);
    }
  });

  it('keeps empty images in classification training with exactly zero box loss', async () => {
    const prediction = { logits: new Float32Array([0, 0]), boxes: new Float32Array([0.5, 0.5, 0.4, 0.4]) };
    const prepared = buildDFineLossTargets(prediction, { labels: new Int32Array(), boxes: new Float32Array() }, { queries: 1, classes: 2 });
    const arrays = makeDFineLossArrays(prepared, { queries: 1, classes: 2 }, 'wasm');
    const [loss, components] = dFineLoss(
      np.array(prediction.logits, { shape: [1, 2] }),
      np.array(prediction.boxes, { shape: [1, 4] }), arrays,
    );
    expect((await loss.data())[0]).toBeCloseTo(0.375 * Math.log(2), 6);
    expect(Array.from(await components.data())).toEqual([expect.closeTo(0.375 * Math.log(2), 6), 0, 0]);
  });

  it('propagates L1 and generalized-IoU gradients through coordinates outside the image', async () => {
    // Equal 1×1 boxes, one shifted right .5: IoU=1/3, GIoU=1/3, L1=.5.
    const prediction = { logits: new Float32Array([0]), boxes: new Float32Array([1, 0.5, 1, 1]) };
    const prepared = buildDFineLossTargets(prediction, {
      labels: new Int32Array([0]), boxes: new Float32Array([0.5, 0.5, 1, 1]),
    }, { queries: 1, classes: 1 });
    const arrays = makeDFineLossArrays(prepared, { queries: 1, classes: 1 }, 'wasm');
    const gradient = grad((boxes: np.Array) => {
      const [loss, parts] = dFineLoss(np.zeros([1, 1]), boxes, tree.ref(arrays));
      parts.dispose();
      return loss;
    })(np.array(prediction.boxes, { shape: [1, 4] }));
    const values = await gradient.data();
    expect(values[0]).toBeCloseTo(5 + 2 * (8 / 9), 5);
    tree.dispose(arrays);
  });

  it('keeps the GIoU gradient finite when matched boxes do not overlap', async () => {
    // Disjoint unit boxes with centers .5 and 2: union2, enclosing2.5, GIoU=-.2.
    // d(1-GIoU)/dx = 2/(2.5²)=.32; weighted total gradient = 5+2*.32.
    const prediction = { logits: new Float32Array([0]), boxes: new Float32Array([2, 0.5, 1, 1]) };
    const prepared = buildDFineLossTargets(prediction, {
      labels: new Int32Array([0]), boxes: new Float32Array([0.5, 0.5, 1, 1]),
    }, { queries: 1, classes: 1 });
    const arrays = makeDFineLossArrays(prepared, { queries: 1, classes: 1 }, 'wasm');
    const gradient = grad((boxes: np.Array) => {
      const [loss, parts] = dFineLoss(np.zeros([1, 1]), boxes, tree.ref(arrays));
      parts.dispose();
      return loss;
    })(np.array(prediction.boxes, { shape: [1, 4] }));
    const values = await gradient.data();
    expect(Array.from(values).every(Number.isFinite)).toBe(true);
    expect(values[0]).toBeCloseTo(5.64, 5);
    tree.dispose(arrays);
  });
});
