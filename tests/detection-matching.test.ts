import { describe, expect, it } from 'vitest';
import {
  boxOverlap, hungarianMatch, linearSumAssignment, validateDetectionTargets,
} from '../src/ml/detection/matching';

describe('D-FINE no-gradient assignment', () => {
  it('finds the global rectangular minimum when greedy nearest choice is wrong', () => {
    // Greedy takes (0,0)=1 then (1,1)=100. The global assignment costs 3.1.
    const pairs = linearSumAssignment(new Float64Array([1, 2, 1.1, 100, 10, 10]), 3, 2);
    expect(Array.from(pairs.predictionIndices)).toEqual([0, 1]);
    expect(Array.from(pairs.targetIndices)).toEqual([1, 0]);
  });

  it('matches more targets than predictions without reusing either side', () => {
    const pairs = linearSumAssignment(new Float64Array([10, 1, 4, 1, 10, 4]), 2, 3);
    expect(Array.from(pairs.predictionIndices)).toEqual([0, 1]);
    expect(Array.from(pairs.targetIndices)).toEqual([1, 0]);
  });

  it('includes both focal classification and box geometry in assignment', () => {
    const pairs = hungarianMatch({
      logits: new Float32Array([-8, 8, 8, -8, -8, -8]),
      boxes: new Float32Array([0.25, 0.25, 0.2, 0.2, 0.75, 0.75, 0.2, 0.2, 0.5, 0.5, 0.1, 0.1]),
    }, {
      labels: new Int32Array([0, 1]),
      boxes: new Float32Array([0.75, 0.75, 0.2, 0.2, 0.25, 0.25, 0.2, 0.2]),
    }, { queries: 3, classes: 2 });
    expect(Array.from(pairs.predictionIndices)).toEqual([0, 1]);
    expect(Array.from(pairs.targetIndices)).toEqual([1, 0]);
  });

  it('handles an empty annotated image with no matched boxes', () => {
    const pairs = hungarianMatch({ logits: new Float32Array([0]), boxes: new Float32Array([0.5, 0.5, 0.2, 0.2]) },
      { labels: new Int32Array(), boxes: new Float32Array() }, { queries: 1, classes: 1 });
    expect(pairs.predictionIndices).toHaveLength(0);
    expect(pairs.targetIndices).toHaveLength(0);
  });

  it('computes generalized IoU including disjoint boxes without clipping predictions', () => {
    expect(boxOverlap([0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1])).toEqual({ iou: 1, giou: 1 });
    const disjoint = boxOverlap([0.25, 0.25, 0.5, 0.5], [1.25, 0.25, 0.5, 0.5]);
    expect(disjoint.iou).toBe(0);
    expect(disjoint.giou).toBeCloseTo(-1 / 3, 14);
  });

  it('rejects invalid annotations and non-finite costs before assignment', () => {
    expect(() => validateDetectionTargets({ labels: new Int32Array([2]), boxes: new Float32Array([0.5, 0.5, 0.2, 0.2]) }, 2)).toThrow();
    expect(() => validateDetectionTargets({ labels: new Int32Array([0]), boxes: new Float32Array([0.5, 0.5, 0, 0.2]) }, 1)).toThrow();
    expect(() => validateDetectionTargets({ labels: new Int32Array([0]), boxes: new Float32Array([0.95, 0.5, 0.2, 0.2]) }, 1)).toThrow();
    expect(() => linearSumAssignment(new Float64Array([NaN]), 1, 1)).toThrow();
    expect(() => linearSumAssignment(new Float64Array([0, 1]), 1, 1)).toThrow();
  });
});
