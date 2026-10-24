import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { unzipSync } from 'fflate';
import { build } from 'vite';
import type { qualifyDetector } from './fixtures/detection-browser';

let bundle: string;
const bundledAssets = new Map<string, { body: string | Uint8Array; contentType: string }>();
test.beforeAll(async () => {
  const built = await build({
    configFile: false, logLevel: 'silent', build: {
      write: false, target: 'es2022', minify: false, assetsInlineLimit: 0,
      rolldownOptions: {
        input: resolve('tests/fixtures/detection-browser.ts'), preserveEntrySignatures: 'strict',
        output: { codeSplitting: false, entryFileNames: 'harness.js' },
      },
    },
  });
  const outputs = Array.isArray(built) ? built : [built];
  const chunks = outputs.flatMap(output => 'output' in output ? output.output : []).filter(output => output.type === 'chunk');
  const entry = chunks.find(chunk => chunk.isEntry);
  expect(entry).toBeDefined();
  expect(entry!.imports).toEqual([]);
  // ORT's optional proxy can refer to this entry; the harness explicitly disables it.
  expect(entry!.dynamicImports.filter(name => name !== entry!.fileName)).toEqual([]);
  bundle = entry!.code;
  for (const output of outputs.flatMap(output => 'output' in output ? output.output : [])) {
    if (output.type === 'asset') bundledAssets.set(output.fileName, {
      body: output.source, contentType: output.fileName.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream',
    });
  }
});

test('bounded D-FINE head updates and full export agree while recording the separate qualification result', async ({ page, context, browser, baseURL }) => {
  const pageUrl = new URL('__detector-qualification/', baseURL!).href;
  const moduleUrl = new URL('harness.js', pageUrl).href;
  const requests: string[] = [], unexpectedRequests: string[] = [], errors: string[] = [];
  context.on('request', request => {
    requests.push(request.url());
    const url = new URL(request.url());
    if (request.method() !== 'GET' || url.origin !== new URL(baseURL!).origin || !url.href.startsWith(baseURL!)) {
      unexpectedRequests.push(`${request.method()} ${request.url()}`);
    }
  });
  page.on('pageerror', error => errors.push(error.message));
  // Failed preparation remains outside public/ and dist/. Only this developer
  // test route exposes its explicitly unqualified artifacts to the browser.
  const modelUrl = new URL('models/dfine-nano/', baseURL!).href;
  const modelDirectory = resolve(process.env.OVB_DETECTOR_ASSETS ?? 'artifacts/p1b/detector-assets/unqualified');
  await page.route(`${modelUrl}**`, async route => {
    const relative = decodeURIComponent(route.request().url().slice(modelUrl.length).split('?')[0]);
    const path = resolve(modelDirectory, relative);
    if (!path.startsWith(modelDirectory + sep)) return route.abort();
    try {
      const body = await readFile(path);
      return route.fulfill({ body, contentType: relative.endsWith('.json') ? 'application/json' : 'application/octet-stream' });
    } catch { return route.fulfill({ status: 404, body: 'Prepare the developer detector investigation assets first.' }); }
  });
  await page.route(moduleUrl, route => route.fulfill({ contentType: 'text/javascript', body: bundle }));
  for (const [name, asset] of bundledAssets) {
    await page.route(new URL(name, pageUrl).href, route => route.fulfill({
      contentType: asset.contentType, body: typeof asset.body === 'string' ? asset.body : Buffer.from(asset.body),
    }));
  }
  await page.route(pageUrl, route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>D-FINE developer qualification</title>' }));
  await page.goto(pageUrl);
  const downloading = page.waitForEvent('download', { timeout: 170_000 });
  const result = await page.evaluate(async ({ moduleUrl, baseUrl }) => {
    const harness = await import(moduleUrl) as { qualifyDetector: typeof qualifyDetector };
    return harness.qualifyDetector(baseUrl);
  }, { moduleUrl, baseUrl: baseURL! });
  const download = await downloading;
  const zip = new Uint8Array(await readFile((await download.path())!)), files = unzipSync(zip);
  await mkdir('artifacts/p1b/browser', { recursive: true });
  for (const name of ['model.onnx', 'report.json', 'labels.json', 'manifest.json']) {
    expect(files[name]?.length).toBeGreaterThan(0);
    await writeFile(`artifacts/p1b/browser/${name}`, files[name]);
  }
  await writeFile('artifacts/p1b/browser/detector-kit.zip', zip);
  await writeFile('artifacts/p1b/browser/network.json', `${JSON.stringify({
    browserVersion: browser.version(), requests, unexpectedRequests, errors,
    harnessSha256: createHash('sha256').update(bundle).digest('hex'),
  }, null, 2)}\n`);
  expect(result.report.environment.crossOriginIsolated).toBe(false);
  expect(['developer-investigation', 'unqualified-intermediate-parity-failure']).toContain(result.report.preparedStatus);
  expect(result.report.training.stepsCompleted).toBe(2);
  expect(result.report.training.metrics.localizationGradientL2).toBeGreaterThan(0);
  expect(result.report.training.metrics.classificationGradientL2).toBeGreaterThan(0);
  expect(result.report.training.metrics.localizationDeltaL2).toBeGreaterThan(0);
  expect(result.report.training.metrics.classificationDeltaL2).toBeGreaterThan(0);
  expect(result.report.training.frozenLqeUnchanged).toBe(true);
  expect(result.report.training.lossAfterTraining).toBeLessThan(result.report.training.lossesBeforeUpdate[0]);
  expect(result.report.training.outputChange.boxes).toBeGreaterThan(0);
  expect(result.report.training.outputChange.logits).toBeGreaterThan(0);
  expect(result.report.artifact.bytes).toBeGreaterThan(14_000_000);
  expect(result.report.artifact.bytes).toBeLessThan(25 * 1024 * 1024);
  for (const gate of ['originalHead', 'export'] as const) {
    const values = result.report.verification[gate];
    for (const [name, comparison] of Object.entries(values)) expect(comparison, `${gate}/${name}`).toMatchObject({ passed: true });
  }
  for (const name of ['logits', 'boxes'] as const) expect(result.report.verification.native[name], `native/${name}`).toMatchObject({ passed: true });
  const qualification = result.report.detectorQualification;
  const nativeFailed = Object.values(result.report.verification.native).some(value => !value.passed);
  expect(qualification.status).toBe(nativeFailed || qualification.preparationFailures.length ? 'FAIL' : 'PASS');
  expect(qualification.releaseReady).toBe(false);
  expect(JSON.parse(new TextDecoder().decode(files['manifest.json'])).detectorQualification).toEqual(qualification);
  test.info().annotations.push({ type: 'Detector forward qualification', description: `${qualification.status}; component checks do not clear G1.` });
  for (const name of ['detect.py', 'README.txt', 'LICENSE.txt', 'TRANSFORMERS_LICENSE.txt', 'NOTICE.txt', 'MODEL_CARD.md']) {
    expect(files[name]?.length).toBeGreaterThan(0);
  }
  expect(unexpectedRequests).toEqual([]);
  expect(errors).toEqual([]);
});
