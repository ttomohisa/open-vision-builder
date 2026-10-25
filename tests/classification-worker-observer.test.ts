import { afterEach, expect, it, vi } from 'vitest';
import { createHash } from 'node:crypto';
import { installWorkerObserver } from '../scripts/classification-worker-observer.mjs';
import { validateWorkerRequest } from '../scripts/classification-real-state.mjs';
import dataValue from './fixtures/classification-real-data.json';
import plan from './fixtures/classification-real-plan.json';

afterEach(() => vi.unstubAllGlobals());

function setup() {
  const forwarded: unknown[] = [];
  class Endpoint extends EventTarget { postMessage(message: unknown) { forwarded.push(message); } }
  const data = structuredClone(dataValue);
  for (const item of data.images) item.pngSha256 = createHash('sha256').update(item.id).digest('hex');
  const host = { Worker: Endpoint, __ovbForwarding: Promise.resolve(), __ovbRequestCapture: Promise.resolve({}),
    __ovbFailure: '', __ovbValidateRequest: async (request: unknown) => validateWorkerRequest(request, data, plan) };
  vi.stubGlobal('window', host); installWorkerObserver();
  const worker = new host.Worker();
  const message = { type: 'run', id: 1, labels: plan.classes, epochs: 40, learningRate: 0.01,
    samples: data.images.filter(x => x.split !== 'tuning').map((x, i) => ({ id: `worker-${i}`, name: `${x.id}.png`,
      label: x.label, split: x.split === 'train' ? 'train' : 'test', file: new Blob([x.id], { type: 'image/png' }) })) };
  return { host, worker, message, forwarded };
}

it('never forwards a malformed actual training request to the Worker', async () => {
  const { host, worker, message, forwarded } = setup();
  message.samples[0].split = 'test';
  worker.postMessage(message);
  await host.__ovbForwarding;
  expect(forwarded).toEqual([]);
  expect(host.__ovbFailure).toMatch(/split/);
});

it('checks encoded hashes before forwarding the unchanged request exactly once', async () => {
  const { host, worker, message, forwarded } = setup();
  worker.postMessage(message);
  expect(forwarded).toEqual([]);
  await host.__ovbForwarding;
  expect(forwarded).toEqual([message]);
  expect(await host.__ovbRequestCapture).toMatchObject({ epochs: 40, learningRate: 0.01, samples: [{ fileSha256: expect.any(String) }, ...Array(49).fill(expect.any(Object))] });
});
