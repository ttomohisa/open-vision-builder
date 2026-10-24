import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { build } from 'vite';
import { applyGridSamplePatch } from '../scripts/patch-jax-onnx-grid-sample.mjs';
import reference from './fixtures/grid-sample-pytorch.json' with { type: 'json' };
import type { runGridSampleCases } from './fixtures/grid-sample-browser';

// Fixed before execution, identical to the Node comparison gate.
const tolerance = {
  output: { atol: 1e-5, rtol: 1e-5 },
  featureGradient: { atol: 2e-5, rtol: 1e-5 },
  gridGradient: { atol: 2e-5, rtol: 1e-5 },
};
const keys = ['output', 'featureGradient', 'gridGradient'] as const;
let bundle: string;

test.beforeAll(async () => {
  applyGridSamplePatch({ check: true });
  const built = await build({
    configFile: false, logLevel: 'silent',
    build: {
      write: false, target: 'es2022', minify: false,
      lib: { entry: resolve('tests/fixtures/grid-sample-browser.ts'), formats: ['es'] },
      rolldownOptions: { output: { codeSplitting: false } },
    },
  });
  const outputs = Array.isArray(built) ? built : [built];
  const chunks = outputs.flatMap(output => 'output' in output ? output.output : [])
    .filter(output => output.type === 'chunk');
  const entry = chunks.find(chunk => chunk.isEntry);
  expect(entry).toBeDefined();
  // Vite may also emit an unused backend entry. The WASM harness itself must
  // be self-contained so every executed byte comes from the local test route.
  expect(entry!.imports).toEqual([]);
  expect(entry!.dynamicImports).toEqual([]);
  bundle = entry!.code;
});

test('browser WASM GridSample agrees with PyTorch forward and both weighted VJPs', async ({ page, context, browser, baseURL }) => {
  const pageUrl = new URL('__grid-sample-regression/', baseURL!).href;
  const moduleUrl = new URL('harness.js', pageUrl).href;
  const requests: string[] = [];
  const unexpectedRequests: string[] = [];
  const errors: string[] = [];
  context.on('request', request => {
    requests.push(request.url());
    if (request.method() !== 'GET' || ![pageUrl, moduleUrl].includes(request.url())) {
      unexpectedRequests.push(`${request.method()} ${request.url()}`);
    }
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    if (route.request().url() === moduleUrl) return route.fulfill({ contentType: 'text/javascript', body: bundle });
    if (route.request().url() === pageUrl) return route.fulfill({
      contentType: 'text/html', body: '<!doctype html><title>GridSample browser regression</title>',
    });
    return route.abort();
  });
  await page.goto(pageUrl);
  const actual = await page.evaluate(async ({ moduleUrl, cases }) => {
    const harness = await import(moduleUrl) as { runGridSampleCases: typeof runGridSampleCases };
    return harness.runGridSampleCases(cases);
  }, { moduleUrl, cases: reference.cases });
  expect(actual.results).toHaveLength(reference.cases.length);
  const results = actual.results.map((result, index) => {
    const fixture = reference.cases[index];
    expect(result.name).toBe(fixture.name);
    expect(result.outputShape).toEqual(fixture.outputShape);
    const comparison = Object.fromEntries(keys.map(key => {
      const values = result.actual[key];
      const expected = fixture.expected[key];
      expect(values).toHaveLength(expected.length);
      expect(values.every(Number.isFinite), `${fixture.name}/${key}: finite`).toBe(true);
      const errors = values.map((value, i) => Math.abs(value - expected[i]));
      const ratios = errors.map((error, i) => error / (tolerance[key].atol + tolerance[key].rtol * Math.abs(expected[i])));
      return [key, { maximumAbsoluteError: Math.max(...errors), maximumToleranceRatio: Math.max(...ratios) }];
    }));
    return { name: result.name, comparison, passed: Object.values(comparison).every(result => result.maximumToleranceRatio <= 1) };
  });
  const report = {
    scope: 'GridSample operator in browser WASM; no complete detector was trained.',
    operatorGate: results.every(result => result.passed) ? 'PASS' : 'FAIL', detectorGates: 'NOT RUN',
    browserVersion: browser.version(), ...actual,
    reference: reference.reference, tolerance, results,
    fixtureSha256: createHash('sha256').update(await readFile('tests/fixtures/grid-sample-pytorch.json')).digest('hex'),
    bundleSha256: createHash('sha256').update(bundle).digest('hex'),
    implementation: applyGridSamplePatch({ check: true }), requests, unexpectedRequests, errors,
  };
  await mkdir('artifacts/p1b', { recursive: true });
  await writeFile('artifacts/p1b/grid-sample-browser.json', `${JSON.stringify(report, null, 2)}\n`);
  expect(results.filter(result => !result.passed)).toEqual([]);
  expect(unexpectedRequests).toEqual([]);
  expect(errors).toEqual([]);
  expect(actual.crossOriginIsolated).toBe(false);
});
