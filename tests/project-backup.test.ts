import { describe, expect, test } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import type { RunSummary } from '../src/protocol';
import { createProjectBackup, parseProjectBackup } from '../src/project/backup';
import type { ProjectSnapshot } from '../src/project/types';

function summary(): RunSummary {
  return {
    epochs: 40, losses: [1, 0.5], accuracy: 1,
    predictions: [{ id: 'eval-1', actual: 1, predicted: 1, scores: [0.1, 0.9] }],
    trainingCount: 1, testCount: 1, durationMs: 123,
    nativeMaxAbsError: 1e-6, exportMaxAbsError: 2e-6, weightDeltaL2: 0.4,
    modelBytes: 12, modelSha256: 'a'.repeat(64), backend: 'wasm',
  };
}

function fakeModelKit() {
  return zipSync({
    'model.onnx': new Uint8Array([8, 1, 2, 3]),
    'labels.json': strToU8('["cat","dog"]'),
    'manifest.json': strToU8('{"schemaVersion":1}'),
    'report.json': strToU8('{"ok":true}'),
  }, { level: 0 });
}

function snapshot(): ProjectSnapshot {
  return {
    labels: ['cat', 'dog'],
    epochs: '40',
    learningRate: '0.01',
    samples: [
      { id: 'train-1', name: 'cat.png', label: 0, split: 'train', synthetic: false, file: new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }) },
      { id: 'eval-1', name: 'dog.jpg', label: 1, split: 'test', synthetic: false, file: new Blob([new Uint8Array([4, 5, 6, 7])], { type: 'image/jpeg' }) },
    ],
    result: { summary: summary(), archive: fakeModelKit() },
  };
}

async function bytes(blob: Blob) {
  return Array.from(new Uint8Array(await blob.arrayBuffer()));
}

describe('project backup', () => {
  test('round-trips project data, image bytes, settings and trained model kit', async () => {
    const original = snapshot();
    const archive = await createProjectBackup(original);
    const files = unzipSync(archive);
    expect(Object.keys(files).sort()).toEqual(['images/eval-1', 'images/train-1', 'project.json', 'result/model-kit.zip']);
    const manifest = JSON.parse(strFromU8(files['project.json']));
    expect(manifest.schemaVersion).toBe(1);
    expect(manifest.task).toBe('classification');
    expect(manifest.model).toBe('mobilenetv4-conv-small');

    const restored = await parseProjectBackup(archive);
    expect(restored.labels).toEqual(original.labels);
    expect(restored.epochs).toBe('40');
    expect(restored.learningRate).toBe('0.01');
    expect(restored.samples.map(item => ({ id: item.id, name: item.name, label: item.label, split: item.split, synthetic: item.synthetic, type: item.file.type })))
      .toEqual(original.samples.map(item => ({ id: item.id, name: item.name, label: item.label, split: item.split, synthetic: item.synthetic, type: item.file.type })));
    expect(await bytes(restored.samples[0].file)).toEqual([1, 2, 3]);
    expect(await bytes(restored.samples[1].file)).toEqual([4, 5, 6, 7]);
    expect(restored.result?.summary).toEqual(original.result?.summary);
    expect(Array.from(restored.result!.archive)).toEqual(Array.from(original.result!.archive));
  });

  test('rejects path traversal before accepting a project', async () => {
    const archive = zipSync({
      'project.json': strToU8(JSON.stringify({ schemaVersion: 1, task: 'classification', model: 'mobilenetv4-conv-small', labels: ['a', 'b'], epochs: '40', learningRate: '0.01', samples: [] })),
      '../outside': new Uint8Array([1]),
    }, { level: 0 });
    await expect(parseProjectBackup(archive)).rejects.toThrow(/path|entry|ZIP/i);
  });

  test('rejects compressed backup entries to bound decompression', async () => {
    const archive = zipSync({ 'project.json': strToU8('{}') }, { level: 6 });
    await expect(parseProjectBackup(archive)).rejects.toThrow(/compress|stored|ZIP/i);
  });

  test('rejects a binary checksum mismatch', async () => {
    const archive = await createProjectBackup(snapshot());
    const files = unzipSync(archive);
    const manifest = JSON.parse(strFromU8(files['project.json']));
    manifest.samples[0].file.sha256 = '0'.repeat(64);
    files['project.json'] = strToU8(JSON.stringify(manifest));
    const tampered = zipSync(files, { level: 0 });
    await expect(parseProjectBackup(tampered)).rejects.toThrow(/checksum|SHA/i);
  });

  test('rejects invalid class references without changing the caller state', async () => {
    const archive = await createProjectBackup(snapshot());
    const files = unzipSync(archive);
    const manifest = JSON.parse(strFromU8(files['project.json']));
    manifest.samples[0].label = 99;
    files['project.json'] = strToU8(JSON.stringify(manifest));
    const tampered = zipSync(files, { level: 0 });
    await expect(parseProjectBackup(tampered)).rejects.toThrow(/label|class/i);
  });

  test('rejects more than 64 images', async () => {
    const base = snapshot();
    base.result = null;
    base.samples = Array.from({ length: 65 }, (_, index) => ({
      id: `item-${index}`, name: `image-${index}.png`, label: index % 2, split: index % 3 ? 'train' as const : 'test' as const,
      synthetic: false, file: new Blob([new Uint8Array([index])], { type: 'image/png' }),
    }));
    await expect(createProjectBackup(base)).rejects.toThrow(/64|image/i);
  });

  test('requires a valid model kit when a trained result is present', async () => {
    const invalid = snapshot();
    invalid.result = { summary: summary(), archive: new Uint8Array([1, 2, 3]) };
    await expect(createProjectBackup(invalid)).rejects.toThrow(/model|kit|ZIP/i);
  });
});
