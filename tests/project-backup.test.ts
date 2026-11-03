import { describe, expect, it } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import {
  exportClassificationProject, importClassificationProject, inspectProjectZip,
} from '../src/project-backup';
import type { ClassificationProject } from '../src/project-backup';

// A tiny public-domain-free synthetic pixel PNG, not a user photograph.
const png = new Uint8Array(Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAAN0lEQVR4nKXMQQ2AMBQFwfkEA2hEB7TVh6c+HBCS7nkzjvOhNjNNzYhIAnobdpSPfg3XsnAvCS+lsw1P37GEkAAAAABJRU5ErkJggg==',
  'base64',
));
function project(): ClassificationProject {
  return {
    labels: ['花', '車'], epochs: 40, learningRate: 0.01,
    syntheticIds: ['a'],
    samples: [
      { id: 'a', name: '一.png', label: 0, split: 'train',
        file: new Blob([png], { type: 'image/png' }) },
      { id: 'b', name: 'two.png', label: 1, split: 'test',
        file: new Blob([png], { type: 'image/png' }) },
    ],
  };
}
async function withModifiedManifest(change: (manifest: Record<string, any>) => void): Promise<Uint8Array> {
  const files = unzipSync(await exportClassificationProject(project()));
  const manifest = JSON.parse(strFromU8(files['manifest.json']));
  change(manifest);
  files['manifest.json'] = strToU8(JSON.stringify(manifest));
  return zipSync(files, { level: 0 });
}

describe('classification project backup', () => {
  it('round-trips original bytes, labels, split, settings and synthetic flag', async () => {
    const data = project();
    const archive = await exportClassificationProject(data);
    inspectProjectZip(archive);
    const restored = await importClassificationProject(archive);
    expect(restored.labels).toEqual(data.labels);
    expect(restored.epochs).toBe(40);
    expect(restored.learningRate).toBe(0.01);
    expect(restored.syntheticIds).toEqual(['a']);
    expect(restored.samples.map(sample => [sample.id, sample.name, sample.label, sample.split]))
      .toEqual(data.samples.map(sample => [sample.id, sample.name, sample.label, sample.split]));
    expect(new Uint8Array(await restored.samples[0].file.arrayBuffer())).toEqual(png);
  });

  it('saves partial datasets with no images for later annotation', async () => {
    const data = project(); data.samples = [];
    const restored = await importClassificationProject(await exportClassificationProject(data));
    expect(restored.samples).toEqual([]);
    expect(restored.labels).toEqual(data.labels);
  });

  it('rejects tampered image contents even when the ZIP container is valid', async () => {
    const files = unzipSync(await exportClassificationProject(project()));
    files['images/000.bin'][25] ^= 1;
    await expect(importClassificationProject(zipSync(files, { level: 0 }))).rejects.toThrow(/欠損|変更|破損/);
  });

  it('rejects missing image, extra file, invalid manifest and unsupported version', async () => {
    const valid = unzipSync(await exportClassificationProject(project()));
    delete valid['images/001.bin'];
    await expect(importClassificationProject(zipSync(valid))).rejects.toThrow();
    const extra = unzipSync(await exportClassificationProject(project()));
    extra['../evil.txt'] = strToU8('no');
    expect(() => inspectProjectZip(zipSync(extra))).toThrow();
    await expect(importClassificationProject(await withModifiedManifest(m => { m.schemaVersion = 2; }))).rejects.toThrow();
    await expect(importClassificationProject(await withModifiedManifest(m => { m.images[0].path = '../evil'; }))).rejects.toThrow();
  });

  it('rejects duplicate sample ids, invalid labels and invalid settings', async () => {
    const data = project();
    data.samples[1].id = data.samples[0].id;
    await expect(exportClassificationProject(data)).rejects.toThrow();
    data.samples[1].id = 'b';
    data.labels = ['花', '花'];
    await expect(exportClassificationProject(data)).rejects.toThrow();
    data.labels = ['花', '車'];
    data.epochs = 300;
    await expect(exportClassificationProject(data)).rejects.toThrow();
    await expect(importClassificationProject(await withModifiedManifest(m => {
      m.images[1].id = m.images[0].id;
    }))).rejects.toThrow();
  });

  it('rejects a declared decompression bomb before inflation', async () => {
    const bytes = await exportClassificationProject(project());
    const corrupt = new Uint8Array(bytes);
    const view = new DataView(corrupt.buffer);
    for (let i = 0; i < corrupt.length - 46; i++) {
      if (view.getUint32(i, true) === 0x02014b50) {
        const nameLen = view.getUint16(i + 28, true);
        const name = strFromU8(corrupt.subarray(i + 46, i + 46 + nameLen));
        if (name === 'images/000.bin') { view.setUint32(i + 24, 100_000_000, true); break; }
      }
    }
    expect(() => inspectProjectZip(corrupt)).toThrow(/サイズ/);
  });
});
