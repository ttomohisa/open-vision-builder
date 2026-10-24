import { describe, expect, it } from 'vitest';
import { runGridSampleProbe } from '../scripts/probe-dfine.mjs';

describe('GridSample through ONNXModel', () => {
  it.each(['esm', 'cjs'] as const)('%s matches independent forward and both weighted VJPs', async (format) => {
    // Removing the raw-floor correction must fail the left/top/corner cases.
    // Nonuniform signed cotangents also catch batch/channel gather and scatter mistakes.
    const report = await runGridSampleProbe({ format });
    const failed = [
      ...report.results.filter((item: { passed: boolean }) => !item.passed),
      ...report.pytorch.results.filter((item: { passed: boolean }) => !item.passed),
    ];
    expect(failed, JSON.stringify(failed, null, 2)).toEqual([]);
    expect(report.operatorGate).toBe('PASS');
    expect(report.detectorGates).toBe('NOT RUN');
    expect(report.results).toHaveLength(6);
    expect(report.pytorch.results).toHaveLength(12);
  });
});
