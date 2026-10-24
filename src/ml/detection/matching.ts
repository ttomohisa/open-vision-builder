/** Normalized [center x, center y, width, height] annotations for one image. */
export interface DetectionTargets {
  labels: Int32Array;
  boxes: Float32Array;
}

export interface MatchedIndices {
  predictionIndices: Int32Array;
  targetIndices: Int32Array;
}

export interface MatchPrediction {
  logits: Float32Array;
  boxes: Float32Array;
}

export interface MatchShape { queries: number; classes: number }

export function validateDetectionTargets(targets: DetectionTargets, classes: number): void {
  if (!Number.isInteger(classes) || classes < 1) throw new Error('Invalid detection class count.');
  if (!(targets.labels instanceof Int32Array) || !(targets.boxes instanceof Float32Array)
    || targets.boxes.length !== targets.labels.length * 4) {
    throw new Error('Detection labels and boxes have incompatible shapes.');
  }
  for (let i = 0; i < targets.labels.length; i++) {
    if (targets.labels[i] < 0 || targets.labels[i] >= classes) throw new Error('Detection label is outside the model class range.');
    const box = targets.boxes.subarray(i * 4, i * 4 + 4);
    if (!Array.from(box).every(Number.isFinite) || box[2] <= 0 || box[3] <= 0
      || box[0] - box[2] / 2 < -1e-6 || box[1] - box[3] / 2 < -1e-6
      || box[0] + box[2] / 2 > 1 + 1e-6 || box[1] + box[3] / 2 > 1 + 1e-6) {
      throw new Error('Target boxes must have positive size and lie within the normalized image.');
    }
  }
}

/** No clipping: D-FINE predictions may extend beyond the image boundary. */
export function boxOverlap(a: ArrayLike<number>, b: ArrayLike<number>): { iou: number; giou: number } {
  const ax1 = a[0] - a[2] / 2; const ay1 = a[1] - a[3] / 2;
  const ax2 = a[0] + a[2] / 2; const ay2 = a[1] + a[3] / 2;
  const bx1 = b[0] - b[2] / 2; const by1 = b[1] - b[3] / 2;
  const bx2 = b[0] + b[2] / 2; const by2 = b[1] + b[3] / 2;
  const intersection = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
    * Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const union = a[2] * a[3] + b[2] * b[3] - intersection;
  const enclosing = (Math.max(ax2, bx2) - Math.min(ax1, bx1))
    * (Math.max(ay2, by2) - Math.min(ay1, by1));
  if (!(union > 0) || !(enclosing > 0)) throw new Error('Degenerate boxes cannot be used for IoU.');
  const iou = intersection / union;
  return { iou, giou: iou - (enclosing - union) / enclosing };
}

/**
 * Rectangular shortest augmenting path assignment. Rows are predictions, columns
 * are targets. This host computation runs outside autodiff, just as the native
 * D-FINE HungarianMatcher is decorated with torch.no_grad().
 */
export function linearSumAssignment(cost: Float64Array, rows: number, columns: number): MatchedIndices {
  if (!Number.isInteger(rows) || !Number.isInteger(columns) || rows < 0 || columns < 0
    || cost.length !== rows * columns || !cost.every(Number.isFinite)) {
    throw new Error('Assignment costs must be a finite rectangular matrix.');
  }
  if (rows === 0 || columns === 0) return { predictionIndices: new Int32Array(), targetIndices: new Int32Array() };
  const transpose = rows > columns;
  const n = Math.min(rows, columns); const m = Math.max(rows, columns);
  const u = new Float64Array(n + 1); const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1); const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    const min = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    let column = 0;
    do {
      used[column] = 1;
      const row = p[column];
      let delta = Infinity; let next = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const originalIndex = transpose ? (j - 1) * columns + row - 1 : (row - 1) * columns + j - 1;
        const reduced = cost[originalIndex] - u[row] - v[j];
        if (reduced < min[j]) { min[j] = reduced; way[j] = column; }
        if (min[j] < delta) { delta = min[j]; next = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else min[j] -= delta;
      }
      column = next;
    } while (p[column] !== 0);
    do {
      const previous = way[column];
      p[column] = p[previous];
      column = previous;
    } while (column !== 0);
  }
  const pairs: [number, number][] = [];
  for (let j = 1; j <= m; j++) {
    if (p[j]) pairs.push(transpose ? [j - 1, p[j] - 1] : [p[j] - 1, j - 1]);
  }
  pairs.sort((a, b) => a[0] - b[0]);
  return {
    predictionIndices: Int32Array.from(pairs, (pair) => pair[0]),
    targetIndices: Int32Array.from(pairs, (pair) => pair[1]),
  };
}

export function validateMatchPrediction(prediction: MatchPrediction, shape: MatchShape): void {
  if (!Number.isInteger(shape.queries) || shape.queries < 1 || !Number.isInteger(shape.classes) || shape.classes < 1
    || !(prediction.logits instanceof Float32Array) || !(prediction.boxes instanceof Float32Array)
    || prediction.logits.length !== shape.queries * shape.classes || prediction.boxes.length !== shape.queries * 4
    || !prediction.logits.every(Number.isFinite) || !prediction.boxes.every(Number.isFinite)) {
    throw new Error('Detection predictions have invalid shapes or non-finite values.');
  }
  for (let i = 0; i < shape.queries; i++) {
    if (prediction.boxes[4 * i + 2] <= 0 || prediction.boxes[4 * i + 3] <= 0) {
      throw new Error('Predicted boxes must have positive width and height.');
    }
  }
}

/** Fixed Nano matching recipe: focal cost 2, L1 cost 5, GIoU cost 2, alpha .25, gamma 2. */
export function hungarianMatch(prediction: MatchPrediction, targets: DetectionTargets, shape: MatchShape): MatchedIndices {
  validateDetectionTargets(targets, shape.classes);
  validateMatchPrediction(prediction, shape);
  const count = targets.labels.length;
  const cost = new Float64Array(shape.queries * count);
  for (let query = 0; query < shape.queries; query++) {
    const predictedBox = prediction.boxes.subarray(query * 4, query * 4 + 4);
    for (let target = 0; target < count; target++) {
      const targetBox = targets.boxes.subarray(target * 4, target * 4 + 4);
      const logit = prediction.logits[query * shape.classes + targets.labels[target]];
      const probability = 1 / (1 + Math.exp(-logit));
      const positive = 0.25 * (1 - probability) ** 2 * -Math.log(probability + 1e-8);
      const negative = 0.75 * probability ** 2 * -Math.log(1 - probability + 1e-8);
      let l1 = 0;
      for (let k = 0; k < 4; k++) l1 += Math.abs(predictedBox[k] - targetBox[k]);
      cost[query * count + target] = 2 * (positive - negative) + 5 * l1 - 2 * boxOverlap(predictedBox, targetBox).giou;
    }
  }
  return linearSumAssignment(cost, shape.queries, count);
}
