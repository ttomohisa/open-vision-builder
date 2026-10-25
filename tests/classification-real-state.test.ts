import { describe, expect, it } from 'vitest';
import { create, toBinary } from '@bufbuild/protobuf';
import { ModelProtoSchema, TensorProtoSchema } from 'onnx-buf';
import { mkdtemp, readFile, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import planValue from './fixtures/classification-real-plan.json';
import dataValue from './fixtures/classification-real-data.json';
import { extractExportedHead, reserveOutput, validatePlan, validateWorkerRun } from '../scripts/classification-real-state.mjs';

describe('fixed real-image classification experiment boundaries', () => {
  const dataHash = planValue.datasetManifestSha256;
  it('binds the fixed recipe and data before accepting a run', () => {
    expect(validatePlan(structuredClone(planValue), dataValue, dataHash)).toHaveLength(50);
    for (const mutate of [
      (p: typeof planValue) => { p.training.epochs = 41; },
      (p: typeof planValue) => { p.training.learningRate = 0.02; },
      (p: typeof planValue) => { p.tolerances.atol = 0.01; },
      (p: typeof planValue) => { p.criteria.minimumEvaluationAccuracy = 0; },
      (p: typeof planValue) => { p.preprocessing.mean[0] = 0; },
    ]) {
      const p = structuredClone(planValue); mutate(p);
      expect(() => validatePlan(p, dataValue, dataHash)).toThrow();
    }
    expect(() => validatePlan(planValue, dataValue, '0'.repeat(64))).toThrow();
  });
  it('rejects cross-split repeated images/creators and unsafe identifiers', () => {
    for (const key of ['id', 'creatorKey', 'flickrId', 'pngSha256'] as const) {
      const d = structuredClone(dataValue); d.images[8][key] = d.images[0][key];
      expect(() => validatePlan(planValue, d, dataHash)).toThrow();
    }
    const d = structuredClone(dataValue); d.images[0].id = '../escape';
    expect(() => validatePlan(planValue, d, dataHash)).toThrow();
  });
  it('checks actual Worker request bytes, labels, splits and the whole loss history', () => {
    const images = dataValue.images.filter(x => x.split !== 'tuning');
    const request = { labels: planValue.classes, epochs: 40, learningRate: 0.01,
      samples: images.map((x, i) => ({ id: `worker-${i}`, imageId: x.id, name: `${x.id}.png`, label: x.label,
        split: x.split === 'train' ? 'train' : 'test', fileSha256: x.pngSha256 })) };
    const summary = { epochs: 40, trainingCount: 30, testCount: 20, losses: Array(40).fill(1.2),
      backend: 'wasm', weightDeltaL2: 0.3, accuracy: 0.2,
      predictions: request.samples.filter(x => x.split === 'test').map(x => ({ id: x.id, actual: x.label, predicted: 0, scores: [0.2, 0.2, 0.2, 0.2, 0.2] })) };
    expect(() => validateWorkerRun(request, summary, dataValue, planValue)).not.toThrow();
    for (const bad of [
      { ...request, samples: request.samples.slice(1) },
      { ...request, samples: request.samples.map((x, i) => i ? x : { ...x, fileSha256: '0'.repeat(64) }) },
      { ...request, samples: request.samples.map((x, i) => i ? x : { ...x, split: 'test' }) },
      { ...request, labels: [...request.labels].reverse() },
    ]) expect(() => validateWorkerRun(bad, summary, dataValue, planValue)).toThrow();
    expect(() => validateWorkerRun(request, { ...summary, losses: Array(39).fill(1) }, dataValue, planValue)).toThrow();
    expect(() => validateWorkerRun(request, { ...summary, losses: [NaN, ...Array(39).fill(1)] }, dataValue, planValue)).toThrow();
  });
  it('refuses to overwrite an earlier result or use a symlink parent', async () => {
    const parent = await mkdtemp(join(tmpdir(), 'ovb-classification-state-'));
    const output = join(parent, 'run'); await reserveOutput(output);
    await writeFile(join(output, 'status.json'), 'previous failure');
    await expect(reserveOutput(output)).rejects.toThrow();
    expect(await readFile(join(output, 'status.json'), 'utf8')).toBe('previous failure');
    await symlink(output, join(parent, 'alias'));
    await expect(reserveOutput(join(parent, 'alias', 'new'))).rejects.toThrow();
  });
  it('extracts feature-major FP32 head bytes and rejects transposed or nonfinite tensors', () => {
    const model = create(ModelProtoSchema, { graph: { initializer: [
      create(TensorProtoSchema, { name: 'ovb.head.weight', dataType: 1, dims: [2n, 3n], rawData: new Uint8Array(new Float32Array([1, 2, 3, 4, 5, 6]).buffer) }),
      create(TensorProtoSchema, { name: 'ovb.head.bias', dataType: 1, dims: [3n], rawData: new Uint8Array(new Float32Array([0.1, 0.2, 0.3]).buffer) }),
    ] } });
    expect(extractExportedHead(toBinary(ModelProtoSchema, model), 3, 2).weights).toEqual([1, 2, 3, 4, 5, 6]);
    model.graph!.initializer[0].dims = [3n, 2n];
    expect(() => extractExportedHead(toBinary(ModelProtoSchema, model), 3, 2)).toThrow();
    model.graph!.initializer[0].dims = [2n, 3n];
    model.graph!.initializer[1].rawData = new Uint8Array(new Float32Array([NaN, 0, 0]).buffer);
    expect(() => extractExportedHead(toBinary(ModelProtoSchema, model), 3, 2)).toThrow();
  });
});
