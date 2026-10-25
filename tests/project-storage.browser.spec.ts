import { test, expect } from '@playwright/test';
import { build } from 'vite';
import { resolve } from 'node:path';
import type { roundTrip, clearAndLoad, corruptObjectAndLoad } from './fixtures/project-storage-browser';

let bundle: string;

test.beforeAll(async () => {
  const built = await build({
    configFile: false,
    logLevel: 'silent',
    build: {
      write: false,
      target: 'es2022',
      minify: false,
      lib: { entry: resolve('tests/fixtures/project-storage-browser.ts'), formats: ['es'] },
      rolldownOptions: { output: { codeSplitting: false } },
    },
  });
  const outputs = Array.isArray(built) ? built : [built];
  const chunks = outputs.flatMap(output => 'output' in output ? output.output : []).filter(output => output.type === 'chunk');
  const entry = chunks.find(chunk => chunk.isEntry);
  expect(entry).toBeDefined();
  expect(entry!.imports).toEqual([]);
  expect(entry!.dynamicImports).toEqual([]);
  bundle = entry!.code;
});

async function openHarness(page: import('@playwright/test').Page, baseURL: string) {
  const pageUrl = new URL('__project-storage/', baseURL).href;
  const moduleUrl = new URL('harness.js', pageUrl).href;
  await page.route('**/*', route => {
    if (route.request().url() === moduleUrl) return route.fulfill({ contentType: 'text/javascript', body: bundle });
    if (route.request().url() === pageUrl) return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Project storage</title>' });
    return route.abort();
  });
  await page.goto(pageUrl);
  return moduleUrl;
}

test('saves manifest in IndexedDB and blobs in OPFS and loads them byte-identically', async ({ page, baseURL }) => {
  const moduleUrl = await openHarness(page, baseURL!);
  const actual = await page.evaluate(async moduleUrl => {
    const harness = await import(moduleUrl) as { roundTrip: typeof roundTrip };
    return harness.roundTrip();
  }, moduleUrl);
  expect(actual).toEqual({
    labels: ['alpha', 'beta'],
    epochs: '23',
    learningRate: '0.02',
    result: null,
    samples: [
      { id: 'train-a', name: 'alpha.png', label: 0, split: 'train', synthetic: false, type: 'image/png', bytes: [1, 3, 5, 7] },
      { id: 'test-b', name: 'beta.webp', label: 1, split: 'test', synthetic: true, type: 'image/webp', bytes: [2, 4, 6, 8, 10] },
    ],
  });
});

test('clear removes the saved manifest and project objects', async ({ page, baseURL }) => {
  const moduleUrl = await openHarness(page, baseURL!);
  const actual = await page.evaluate(async moduleUrl => {
    const harness = await import(moduleUrl) as { clearAndLoad: typeof clearAndLoad };
    return harness.clearAndLoad();
  }, moduleUrl);
  expect(actual).toBeNull();
});

test('missing content-addressed object fails restore instead of returning partial data', async ({ page, baseURL }) => {
  const moduleUrl = await openHarness(page, baseURL!);
  const actual = await page.evaluate(async moduleUrl => {
    const harness = await import(moduleUrl) as { corruptObjectAndLoad: typeof corruptObjectAndLoad };
    return harness.corruptObjectAndLoad();
  }, moduleUrl);
  expect(actual.removed).toMatch(/^[0-9a-f]{64}$/);
  expect(actual.error).toMatch(/missing|object|saved project/i);
});
