import { lax, numpy as np } from '@jax-js/jax';
import {
  boxOverlap, hungarianMatch,
  type DetectionTargets, type MatchedIndices, type MatchPrediction, type MatchShape,
} from './matching';

export interface DFineLossTargets {
  targetScores: Float32Array;
  vflWeights: Float32Array;
  matchedBoxes: Float32Array;
  matches: MatchedIndices;
  normalizer: number;
}

export interface DFineLossArrays {
  targetScores: np.Array;
  vflWeights: np.Array;
  matchedBoxes: np.Array;
  predictionIndices: np.Array;
  normalizer: np.Array;
}

/**
 * Final-output recipe adapted from native D-FINE criterion at 956d1709314c2c6a4df6f34de232054578a7449f.
 * Matching, IoU target scores and probability-dependent VFL weights are snapshots
 * outside autodiff. This explicitly preserves the native detach operations.
 * Auxiliary, denoising, FGL and DDF losses are outside this bounded recipe.
 */
export function buildDFineLossTargets(
  prediction: MatchPrediction, targets: DetectionTargets, shape: MatchShape,
): DFineLossTargets {
  const matches = hungarianMatch(prediction, targets, shape);
  const targetScores = new Float32Array(shape.queries * shape.classes);
  const vflWeights = new Float32Array(shape.queries * shape.classes);
  for (let i = 0; i < prediction.logits.length; i++) {
    const score = 1 / (1 + Math.exp(-prediction.logits[i]));
    vflWeights[i] = 0.75 * score ** 2;
  }
  const matchedBoxes = new Float32Array(matches.predictionIndices.length * 4);
  for (let i = 0; i < matches.predictionIndices.length; i++) {
    const query = matches.predictionIndices[i]; const target = matches.targetIndices[i];
    const targetBox = targets.boxes.subarray(target * 4, target * 4 + 4);
    const iou = boxOverlap(prediction.boxes.subarray(query * 4, query * 4 + 4), targetBox).iou;
    const index = query * shape.classes + targets.labels[target];
    targetScores[index] = iou;
    vflWeights[index] = iou;
    matchedBoxes.set(targetBox, i * 4);
  }
  return { targetScores, vflWeights, matchedBoxes, matches, normalizer: Math.max(targets.labels.length, 1) };
}

export function makeDFineLossArrays(targets: DFineLossTargets, shape: MatchShape, device: 'wasm'): DFineLossArrays {
  const options = { dtype: np.float32, device };
  return {
    targetScores: np.array(new Float32Array(targets.targetScores), { ...options, shape: [shape.queries, shape.classes] }),
    vflWeights: np.array(new Float32Array(targets.vflWeights), { ...options, shape: [shape.queries, shape.classes] }),
    // jax 0.1.27's typed-array constructor reads data[0] for empty arrays.
    matchedBoxes: targets.matchedBoxes.length
      ? np.array(new Float32Array(targets.matchedBoxes), { ...options, shape: [targets.matches.predictionIndices.length, 4] })
      : np.zeros([0, 4], options),
    predictionIndices: targets.matches.predictionIndices.length
      ? np.array(new Int32Array(targets.matches.predictionIndices), { dtype: np.int32, device })
      : np.zeros([0], { dtype: np.int32, device }),
    normalizer: np.array(targets.normalizer, options),
  };
}

function corners(boxes: np.Array): [np.Array, np.Array] {
  const centers = boxes.ref.slice([], [0, 2]);
  const halfSizes = boxes.slice([], [2, 4]).div(2);
  return [centers.ref.sub(halfSizes.ref), centers.add(halfSizes)];
}

function area(sizes: np.Array): np.Array {
  // jax 0.1.27 prod differentiates by division and yields NaN at zero width.
  // Explicit width*height has the native finite derivative for disjoint boxes.
  return sizes.ref.slice([], [0, 1]).mul(sizes.slice([], [1, 2])).reshape([-1]);
}

/** Aligned GIoU, with the same unclipped center/size box convention as native D-FINE. */
function generalizedIoU(predicted: np.Array, targets: np.Array): np.Array {
  const predictedArea = area(predicted.ref.slice([], [2, 4]));
  const targetArea = area(targets.ref.slice([], [2, 4]));
  const [p0, p1] = corners(predicted);
  const [t0, t1] = corners(targets);
  const intersection = area(np.maximum(np.minimum(p1.ref, t1.ref).sub(np.maximum(p0.ref, t0.ref)), 0));
  const union = predictedArea.add(targetArea).sub(intersection.ref);
  const enclosing = area(np.maximum(p1, t1).sub(np.minimum(p0, t0)));
  return intersection.div(union.ref).sub(enclosing.ref.sub(union).div(enclosing));
}

/** Consumes every input reference. Returns [total, [unweighted VFL, L1, GIoU]]. */
export function dFineLoss(logits: np.Array, boxes: np.Array, targets: DFineLossArrays): [np.Array, np.Array] {
  // jax 0.1.27 softplus evaluates exp(logits), which overflows in FP32.
  // A detached log-sum-exp shift keeps both exponentials <=1 and preserves the
  // softplus derivative sigmoid(logits), including exactly .5 at zero.
  const shift = lax.stopGradient(np.maximum(logits.ref, 0));
  const softplus = np.log(np.exp(logits.ref.sub(shift.ref)).add(np.exp(shift.ref.neg()))).add(shift);
  const vfl = softplus.sub(logits.mul(targets.targetScores)).mul(targets.vflWeights)
    .sum().div(targets.normalizer.ref);
  let l1: np.Array; let giou: np.Array;
  if (targets.predictionIndices.shape[0] === 0) {
    l1 = boxes.sum().mul(0);
    giou = l1.ref;
    targets.predictionIndices.dispose();
    targets.matchedBoxes.dispose();
    targets.normalizer.dispose();
  } else {
    const matched = np.take(boxes, targets.predictionIndices, 0);
    const difference = matched.ref.sub(targets.matchedBoxes.ref);
    // Native PyTorch L1 chooses zero at the kink; jax abs chooses +1.
    l1 = np.where(np.equal(difference.ref, 0), 0, np.abs(difference)).sum().div(targets.normalizer.ref);
    giou = generalizedIoU(matched, targets.matchedBoxes).neg().add(1).sum().div(targets.normalizer);
  }
  const total = vfl.ref.add(l1.ref.mul(5)).add(giou.ref.mul(2));
  return [total, np.stack([vfl, l1, giou])];
}
