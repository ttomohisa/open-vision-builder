import type { ProjectSnapshot } from '../../src/project/types';
import { clearProject, loadProject, saveProject } from '../../src/project/storage';

function snapshot(): ProjectSnapshot {
  return {
    labels: ['alpha', 'beta'],
    epochs: '23',
    learningRate: '0.02',
    samples: [
      { id: 'train-a', name: 'alpha.png', label: 0, split: 'train', synthetic: false, file: new Blob([new Uint8Array([1, 3, 5, 7])], { type: 'image/png' }) },
      { id: 'test-b', name: 'beta.webp', label: 1, split: 'test', synthetic: true, file: new Blob([new Uint8Array([2, 4, 6, 8, 10])], { type: 'image/webp' }) },
    ],
    result: null,
  };
}

async function sampleBytes(project: ProjectSnapshot | null) {
  if (!project) return null;
  return Promise.all(project.samples.map(async sample => ({
    id: sample.id, name: sample.name, label: sample.label, split: sample.split, synthetic: sample.synthetic,
    type: sample.file.type, bytes: Array.from(new Uint8Array(await sample.file.arrayBuffer())),
  })));
}

export async function roundTrip() {
  await clearProject();
  const value = snapshot();
  await saveProject(value);
  const loaded = await loadProject();
  return loaded ? {
    labels: loaded.labels, epochs: loaded.epochs, learningRate: loaded.learningRate,
    samples: await sampleBytes(loaded), result: loaded.result,
  } : null;
}

export async function clearAndLoad() {
  await clearProject();
  await saveProject(snapshot());
  await clearProject();
  return loadProject();
}

export async function corruptObjectAndLoad() {
  await clearProject();
  await saveProject(snapshot());
  const root = await navigator.storage.getDirectory();
  const projects = await root.getDirectoryHandle('projects');
  const project = await projects.getDirectoryHandle('classification-default');
  const objects = await project.getDirectoryHandle('objects');
  let removed = '';
  for await (const [name] of objects.entries()) {
    removed = name;
    await objects.removeEntry(name);
    break;
  }
  if (!removed) throw new Error('No object was written');
  try {
    await loadProject();
    return { removed, error: '' };
  } catch (error) {
    return { removed, error: error instanceof Error ? error.message : String(error) };
  }
}
