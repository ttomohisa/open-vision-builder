// Developer experiment sequencing, independent of model/runtime arithmetic.
export function remainingTimeoutMs(deadline, now) {
  const remaining = Math.floor(deadline - now);
  if (!Number.isSafeInteger(remaining) || remaining < 1) throw new Error('Experiment runtime budget exhausted');
  return remaining;
}

export function validateImpactPlan(plan, manifestSha256) {
  if (!plan || plan.schemaVersion !== 1 || plan.id !== 'dfine-coco-own-prefix-impact-v1') throw new Error('Unknown impact plan');
  if (!/^[a-f0-9]{64}$/.test(manifestSha256) || plan.datasetManifestSha256 !== manifestSha256) throw new Error('Dataset digest is not bound');
  const expected = {
    classes: 80, trainable: 'final-pretrained-bbox-and-class-heads', frozen: 'prefix-and-LQE',
    loss: 'VFL + 5 * L1 + 2 * GIoU', optimizer: 'SGD', learningRate: 0.0001,
    momentum: 0, weightDecay: 0, epochs: 3, steps: 12, order: 'manifest-train-order-repeated',
    randomness: 'none; eval prefix; no augmentation or shuffle',
  };
  if (!plan.recipe || Object.keys(plan.recipe).length !== Object.keys(expected).length ||
      Object.entries(expected).some(([key, value]) => plan.recipe[key] !== value)) throw new Error('Changed impact recipe');
  if (JSON.stringify(plan.controls?.checkpoints) !== '[0,6,12]' || plan.controls?.image !== 'first-training-image' ||
      plan.controls?.tolerances !== 'detector-training-tolerances.json') throw new Error('Changed impact controls');
  const criteria = {
    minimumTrainLossReduction: 0.01, maximumOwnLossRelativeGap: 0.01,
    maximumEvaluationAp50Gap: 0.01, maximumEvaluationAp50Drop: 0.02,
    exportLogits: { atol: 0.0002, rtol: 0.0001 }, exportBoxes: { atol: 0.00002, rtol: 0.0001 },
  };
  if (!plan.criteria || Object.keys(plan.criteria).length !== Object.keys(criteria).length ||
      Object.entries(criteria).some(([key, value]) => typeof value === 'number' ? plan.criteria[key] !== value :
        !plan.criteria[key] || Object.keys(plan.criteria[key]).length !== 2 || plan.criteria[key].atol !== value.atol || plan.criteria[key].rtol !== value.rtol)) {
    throw new Error('Changed impact criteria');
  }
  const metrics = { scoreThreshold: 0.05, maxDetections: 100, iouThreshold: 0.5, recallPoints: 101, nms: false };
  if (!plan.metrics || Object.keys(plan.metrics).length !== Object.keys(metrics).length ||
      Object.entries(metrics).some(([key, value]) => plan.metrics[key] !== value)) throw new Error('Changed impact metrics');
  if (plan.splits?.train !== 4 || plan.splits?.tuning !== 2 || plan.splits?.evaluation !== 4 ||
      plan.budget?.completedExperiments !== 1 || plan.budget?.runtimeMinutes !== 15 || plan.budget?.automaticRetries !== 0 ||
      plan.detectorQualification !== 'FAIL' || plan.releaseReady !== false) throw new Error('Changed impact scope/budget');
}

export class ImpactProgress {
  #ids; #steps = 0; #controls = []; #status = 'running'; #reason;
  constructor(trainIds) {
    if (!Array.isArray(trainIds) || trainIds.length !== 4 || new Set(trainIds).size !== 4 ||
        trainIds.some(id => typeof id !== 'string' || !id)) throw new Error('Expected four distinct training images');
    this.#ids = [...trainIds];
  }
  #running() { if (this.#status !== 'running') throw new Error(`Experiment ${this.#status}`); }
  recordControl(step, passed) {
    this.#running();
    if (step !== this.#steps || ![0, 6, 12].includes(step) || this.#controls.includes(step) || typeof passed !== 'boolean') {
      throw new Error('Unexpected control order');
    }
    if (!passed) { this.stop(`same-feature-control-failure-at-${step}`); return; }
    this.#controls.push(step);
  }
  acceptStep(step, id) {
    this.#running();
    if (step !== this.#steps || step >= 12 || id !== this.#ids[step % 4]) throw new Error('Unexpected training order');
    if ([0, 6].includes(step) && !this.#controls.includes(step)) throw new Error('Successful control required before update');
    this.#steps++;
  }
  finish() {
    this.#running();
    if (this.#steps !== 12 || !this.#controls.includes(12)) throw new Error('Final control and twelve steps required');
    this.#status = 'completed';
  }
  stop(reason) {
    if (this.#status === 'completed') throw new Error('Experiment completed');
    this.#status = 'stopped'; this.#reason = String(reason);
  }
  snapshot() {
    return { status: this.#status, completedSteps: this.#steps, controls: [...this.#controls], ...(this.#reason ? { reason: this.#reason } : {}) };
  }
}
