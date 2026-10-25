import { describe, expect, it } from 'vitest';
import { ImpactProgress, validateImpactPlan } from '../scripts/detector-impact-state.mjs';
import plan from './fixtures/detector-impact-plan.json';

describe('bounded detector experiment controller', () => {
  it('refuses updates before each successful same-feature control', () => {
    const state = new ImpactProgress(['a', 'b', 'c', 'd']);
    expect(() => state.acceptStep(0, 'a')).toThrow(/control/);
    state.recordControl(0, true);
    for (let i = 0; i < 6; i++) state.acceptStep(i, ['a', 'b', 'c', 'd'][i % 4]);
    expect(() => state.acceptStep(6, 'c')).toThrow(/control/);
    state.recordControl(6, false);
    expect(() => state.recordControl(6, true)).toThrow(/stopped/);
    expect(() => state.acceptStep(6, 'c')).toThrow(/stopped/);
    expect(state.snapshot()).toMatchObject({ status: 'stopped', completedSteps: 6 });
  });

  it('rejects skipped, duplicated or reordered updates and requires the final control', () => {
    const state = new ImpactProgress(['a', 'b', 'c', 'd']);
    state.recordControl(0, true);
    expect(() => state.acceptStep(1, 'b')).toThrow(/order/);
    expect(() => state.acceptStep(0, 'b')).toThrow(/order/);
    for (let i = 0; i < 12; i++) {
      if (i === 6) state.recordControl(6, true);
      state.acceptStep(i, ['a', 'b', 'c', 'd'][i % 4]);
    }
    expect(() => state.finish()).toThrow(/control/);
    state.recordControl(12, true);
    state.finish();
    expect(state.snapshot()).toMatchObject({ status: 'completed', completedSteps: 12 });
    expect(() => state.acceptStep(12, 'a')).toThrow(/completed/);
  });

  it('refuses an unbound plan or a changed experiment schedule', () => {
    const bound = { ...plan, datasetManifestSha256: 'a'.repeat(64) };
    expect(() => validateImpactPlan(bound, 'a'.repeat(64))).not.toThrow();
    expect(() => validateImpactPlan({ ...bound, datasetManifestSha256: null }, 'a'.repeat(64))).toThrow(/digest/);
    expect(() => validateImpactPlan(bound, 'b'.repeat(64))).toThrow(/digest/);
    expect(() => validateImpactPlan({ ...bound, recipe: { ...bound.recipe, steps: 24 } }, 'a'.repeat(64))).toThrow(/recipe/);
    expect(() => validateImpactPlan({ ...bound, controls: { ...bound.controls, checkpoints: [0, 12] } }, 'a'.repeat(64))).toThrow(/control/);
  });

  it('rejects changing the learning or quality decision conditions before a run', () => {
    const bound = { ...plan, datasetManifestSha256: 'a'.repeat(64) };
    for (const key of ['minimumTrainLossReduction', 'maximumOwnLossRelativeGap', 'maximumEvaluationAp50Gap', 'maximumEvaluationAp50Drop']) {
      expect(() => validateImpactPlan({ ...bound, criteria: { ...bound.criteria, [key]: 1 } }, 'a'.repeat(64))).toThrow(/criteria/);
    }
    expect(() => validateImpactPlan({ ...bound, metrics: { ...bound.metrics, scoreThreshold: 0.5 } }, 'a'.repeat(64))).toThrow(/metrics/);
  });
});
