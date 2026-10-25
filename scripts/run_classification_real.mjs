// Bounded developer-only experiment; user-facing training stays entirely in App/Worker.
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { createServer } from 'node:http';
import { lstat, mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs, promisify } from 'node:util';
import { build } from 'vite';
import { chromium } from '@playwright/test';
import { unzipSync } from 'fflate';
import { extractExportedHead, reserveOutput, validatePlan, validateWorkerRequest, validateWorkerRun } from './classification-real-state.mjs';
import { installWorkerObserver } from './classification-worker-observer.mjs';
import { remainingTimeoutMs } from './detector-impact-state.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const { values: args } = parseArgs({ options: {
  check: { type: 'boolean', default: false }, chromium: { type: 'string' },
  python: { type: 'string', default: resolve(root, '.venv/bin/python') },
  images: { type: 'string', default: resolve(root, 'artifacts/p1e/data/images') },
  output: { type: 'string', default: resolve(root, 'artifacts/p1e/run') },
  cache: { type: 'string', default: resolve(root, '.venv/model-source-cache') },
} });
const planPath = resolve(root, 'tests/fixtures/classification-real-plan.json');
const dataPath = resolve(root, 'tests/fixtures/classification-real-data.json');
const sourceModel = resolve(root, 'public/models/mobilenetv4-small/backbone.onnx');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const exec = promisify(execFile);
const json = async path => JSON.parse(await readFile(path, 'utf8'));
const mime = path => path.endsWith('.js') || path.endsWith('.mjs') ? 'text/javascript' :
  path.endsWith('.json') ? 'application/json' : path.endsWith('.wasm') ? 'application/wasm' :
  path.endsWith('.html') ? 'text/html' : path.endsWith('.css') ? 'text/css' :
  path.endsWith('.png') ? 'image/png' : path.endsWith('.svg') ? 'image/svg+xml' : 'application/octet-stream';

async function writeJson(path, value) {
  const temporary = `${path}.part`;
  for (const candidate of [path, temporary]) {
    const info = await lstat(candidate).catch(error => error.code === 'ENOENT' ? null : Promise.reject(error));
    if (info?.isSymbolicLink()) throw new Error('Refusing symlink report');
  }
  await writeFile(temporary, JSON.stringify(value) + '\n'); await rename(temporary, path);
}

async function main() {
  const [planBytes, dataBytes] = await Promise.all([readFile(planPath), readFile(dataPath)]);
  const plan = JSON.parse(planBytes), data = JSON.parse(dataBytes);
  const planSha256 = sha(planBytes), datasetManifestSha256 = sha(dataBytes);
  const images = validatePlan(plan, data, datasetManifestSha256);
  const output = resolve(args.output), prefix = process.env.OVB_BASE ?? '/open-vision-builder/';
  if (!/^\/[a-zA-Z0-9_/-]+\/$/.test(prefix) || prefix.includes('//')) throw new Error('Invalid static base path');
  const prepared = await json(resolve(dirname(sourceModel), 'model.json'));
  const sourceBytes = await readFile(sourceModel);
  if (sha(sourceBytes) !== plan.model.onnxSha256 || prepared.artifact.sha256 !== plan.model.onnxSha256 ||
      prepared.source.sha256 !== plan.model.checkpointSha256 || prepared.source.revision !== plan.model.sourceRevision) throw new Error('Prepared model differs from fixed plan');
  if (sha(await readFile(resolve(root, 'dist/models/mobilenetv4-small/backbone.onnx'))) !== plan.model.onnxSha256) throw new Error('Build contains a different source model; rebuild first');
  const resources = new Map();
  function add(path, bytes, type = mime(path)) { resources.set(path, { bytes, type }); }
  async function addDist(directory, relative = '') {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name), name = `${relative}${entry.name}`;
      if (entry.isSymbolicLink()) throw new Error('Unexpected symlink in static build');
      if (entry.isDirectory()) await addDist(path, `${name}/`);
      else if (entry.isFile()) add(`${prefix}${name}`, await readFile(path));
    }
  }
  await addDist(resolve(root, 'dist'));
  const index = resources.get(`${prefix}index.html`);
  if (!index) throw new Error('Missing built App');
  resources.set(prefix, index);
  add('/favicon.ico', Buffer.alloc(0), 'image/x-icon');
  for (const item of images) {
    const bytes = await readFile(resolve(args.images, `${item.id}.png`));
    if (sha(bytes) !== item.pngSha256 || bytes.length !== item.pngBytes) throw new Error(`Changed PNG: ${item.id}`);
    add(`${prefix}__classification/images/${item.id}.png`, bytes, 'image/png');
  }
  const built = await build({ configFile: false, logLevel: 'silent', build: {
    write: false, target: 'es2022', minify: false, assetsInlineLimit: 0,
    rolldownOptions: { input: resolve(root, 'tests/fixtures/classification-real-browser.ts'), preserveEntrySignatures: 'strict',
      output: { codeSplitting: false, entryFileNames: 'harness.js' } },
  } });
  const bundled = (Array.isArray(built) ? built : [built]).flatMap(item => item.output ?? []);
  const entry = bundled.find(item => item.type === 'chunk' && item.isEntry);
  if (!entry || entry.imports.length || entry.dynamicImports.some(name => name !== entry.fileName)) throw new Error('Unexpected measurement bundle dependency');
  for (const item of bundled) add(`${prefix}__classification/${item.fileName}`, item.type === 'chunk' ? item.code : item.source);
  add(`${prefix}__classification/`, '<!doctype html><title>Classification verification</title><link rel="icon" href="data:,">', 'text/html');

  const requests = [], unexpectedRequests = [], pageErrors = [];
  const status = { schemaVersion: 1, status: 'running', passed: false, releaseReady: false,
    scope: plan.scope, planSha256, datasetManifestSha256, sourceModelSha256: plan.model.onnxSha256 };
  let server, browser, timer, active = false, started = 0;
  const save = async () => {
    status.durationSeconds = (performance.now() - started) / 1000;
    await writeJson(resolve(output, 'status.json'), status);
  };
  try {
    server = createServer((request, response) => {
      const path = request.url?.split('?')[0] ?? '', resource = resources.get(path);
      if (request.method !== 'GET' || !resource) {
        unexpectedRequests.push(`${request.method} ${path}`); response.writeHead(404).end(); return;
      }
      response.writeHead(200, { 'Content-Type': resource.type, 'Cache-Control': 'no-store' }).end(resource.bytes);
    });
    await new Promise((done, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', done); });
    const origin = `http://127.0.0.1:${server.address().port}`, baseUrl = `${origin}${prefix}`;
    const moduleUrl = `${baseUrl}__classification/harness.js`, runtimeUrl = `${baseUrl}runtime/onnxruntime/`;
    browser = await chromium.launch({ headless: true, ...(args.chromium ? { executablePath: args.chromium } : {}) });
    async function context(phase) {
      const ctx = await browser.newContext({ acceptDownloads: true });
      await ctx.route('**/*', async route => {
        const request = route.request(), url = new URL(request.url());
        const freshAllowed = url.pathname.startsWith(`${prefix}__classification/harness`) ||
          url.pathname.startsWith(`${prefix}runtime/onnxruntime/`) || url.pathname === `${prefix}__classification/` ||
          url.pathname === `${prefix}__classification/model.onnx` || url.pathname.startsWith(`${prefix}__classification/inputs/`);
        if (request.method() !== 'GET' || url.origin !== origin || !resources.has(url.pathname) || (phase === 'fresh' && !freshAllowed)) {
          unexpectedRequests.push(`${phase}: ${request.method()} ${url.href}`); await route.abort(); return;
        }
        requests.push({ phase, method: request.method(), path: url.pathname }); await route.continue();
      });
      ctx.on('page', page => page.on('pageerror', error => pageErrors.push(error.message)));
      return ctx;
    }
    const training = await context('training'), page = await training.newPage();
    if (args.check) {
      await page.goto(`${baseUrl}__classification/`);
      await page.evaluate(async url => { await import(url); }, moduleUrl);
      if (unexpectedRequests.length || pageErrors.length) throw new Error('Preflight network/page failure');
      console.log(JSON.stringify({ status: 'preflight-passed', browser: browser.version(), planSha256, datasetManifestSha256,
        sourceModelSha256: plan.model.onnxSha256, images: images.length,
        note: 'Fixed bytes, bundle and browser module import only. No image model inference or training.' }));
      return;
    }
    await reserveOutput(output); active = true; started = performance.now();
    const deadline = started + plan.budget.maxRuntimeSeconds * 1000;
    status.startedAt = new Date().toISOString(); status.phase = 'app-training';
    status.environment = { node: process.version, browser: browser.version(), platform: process.platform, architecture: process.arch };
    status.harnessSha256 = sha(entry.code); status.codeSha256 = {};
    for (const file of ['src/App.tsx', 'src/worker.ts', 'src/ml/head.ts', 'src/ml/onnx.ts', 'src/ml/decode.ts', 'src/ml/preprocess.ts',
      'scripts/run_classification_real.mjs', 'scripts/classification-real-state.mjs', 'scripts/classification-worker-observer.mjs', 'scripts/detector-impact-state.mjs',
      'scripts/classification_real_reference.py', 'tests/fixtures/classification-real-browser.ts']) status.codeSha256[file] = sha(await readFile(resolve(root, file)));
    await save(); await mkdir(resolve(output, 'inputs'));
    timer = setTimeout(() => browser.close().catch(() => {}), remainingTimeoutMs(deadline, performance.now()));
    // Observe the actual production Worker messages; never replace its computation.
    await page.exposeFunction('__ovbValidateRequest', request => {
      validateWorkerRequest(request, data, plan);
      return true;
    });
    await page.addInitScript(installWorkerObserver);
    await page.goto(baseUrl);
    if (await page.evaluate(() => crossOriginIsolated)) throw new Error('Experiment requires nonisolated static hosting');
    for (let i = 2; i < plan.classes.length; i++) await page.getByRole('button', { name: 'クラスを追加' }).click();
    for (let label = 0; label < plan.classes.length; label++) {
      await page.getByLabel(`クラス${label + 1}の名前`, { exact: true }).fill(plan.classes[label]);
      for (const split of ['train', 'evaluation']) {
        const selected = images.filter(item => item.label === label && item.split === split);
        const title = split === 'train' ? '学習用' : '評価用';
        await page.getByLabel(`クラス${label + 1}の${title}画像を追加`, { exact: true }).setInputFiles(selected.map(item => resolve(args.images, `${item.id}.png`)));
        await page.waitForFunction(({ name, count }) => Array.from(document.querySelectorAll('ul')).find(list => list.getAttribute('aria-label') === name)?.children.length === count,
          { name: `${plan.classes[label]}の${title}画像`, count: selected.length }, { timeout: remainingTimeoutMs(deadline, performance.now()) });
      }
    }
    await page.getByRole('button', { name: '学習してONNXを作成' }).click();
    await page.waitForFunction(() => window.__ovbResult || window.__ovbFailure, null, { timeout: remainingTimeoutMs(deadline, performance.now()) });
    const observed = await page.evaluate(async () => ({ result: window.__ovbResult, error: window.__ovbFailure,
      request: await window.__ovbRequestCapture, progress: window.__ovbProgress }));
    await writeJson(resolve(output, 'worker-observation.json'), observed);
    if (observed.error || !observed.result) throw new Error(observed.error ?? 'Worker returned no result');
    validateWorkerRun(observed.request, observed.result.summary, data, plan);
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'モデル一式をダウンロード' }).click();
    const download = await downloadPromise; await download.saveAs(resolve(output, 'model-kit.zip'));
    const archive = unzipSync(await readFile(resolve(output, 'model-kit.zip')));
    for (const name of ['model.onnx', 'labels.json', 'manifest.json', 'report.json', 'LICENSE.txt', 'NOTICE.txt', 'MODEL_CARD.md', 'predict.py', 'README.txt']) {
      if (!archive[name]?.length) throw new Error(`Incomplete kit: ${name}`);
      await writeFile(resolve(output, name), archive[name]);
    }
    const modelBytes = archive['model.onnx'], modelSha256 = sha(modelBytes);
    if (modelSha256 !== observed.result.summary.modelSha256 || modelBytes.length !== observed.result.summary.modelBytes ||
        JSON.stringify(JSON.parse(new TextDecoder().decode(archive['labels.json']))) !== JSON.stringify(plan.classes)) throw new Error('Downloaded artifact identity mismatch');
    const head = extractExportedHead(modelBytes, 5);
    const trace = { schemaVersion: 1, planSha256, datasetManifestSha256, sourceModelSha256: plan.model.onnxSha256,
      modelSha256, exportedModelBytes: modelBytes.length, labels: plan.classes, request: observed.request, summary: observed.result.summary, head, images: [] };
    await page.screenshot({ path: resolve(output, 'app-result.png'), fullPage: true });
    status.phase = 'supplemental-browser-measurement'; await save();
    const helper = await training.newPage(); await helper.goto(`${baseUrl}__classification/`);
    await helper.evaluate(async url => { window.realVision = await import(url); }, moduleUrl);
    await helper.evaluate(async values => window.realVision.initializeSource(...values),
      [`${baseUrl}models/mobilenetv4-small/backbone.onnx`, plan.model.onnxSha256, runtimeUrl, head]);
    for (const item of images) {
      const result = await helper.evaluate(async values => window.realVision.collectImage(...values),
        [`${baseUrl}__classification/images/${item.id}.png`, item.pngSha256, item.rgbaSha256]);
      const input = Buffer.from(result.inputBase64, 'base64');
      if (input.length !== 3 * 224 * 224 * 4 || sha(input) !== result.inputSha256) throw new Error('Input readback mismatch');
      await writeFile(resolve(output, 'inputs', `${item.id}.f32`), input);
      add(`${prefix}__classification/inputs/${item.id}.f32`, input);
      delete result.inputBase64; trace.images.push({ id: item.id, label: item.label, split: item.split, ...result });
      await writeJson(resolve(output, 'browser.json'), trace);
    }
    await helper.evaluate(async () => window.realVision.close()); await training.close();
    status.phase = 'fresh-browser-export'; await save();
    add(`${prefix}__classification/model.onnx`, modelBytes);
    const fresh = await context('fresh'), freshPage = await fresh.newPage();
    await freshPage.goto(`${baseUrl}__classification/`);
    await freshPage.evaluate(async url => { window.realVision = await import(url); }, moduleUrl);
    await freshPage.evaluate(async values => window.realVision.initializeExport(...values),
      [`${baseUrl}__classification/model.onnx`, modelSha256, runtimeUrl]);
    const freshTrace = { schemaVersion: 1, planSha256, datasetManifestSha256, modelSha256, images: [] };
    for (const item of trace.images) {
      const logits = await freshPage.evaluate(async values => window.realVision.inferExport(...values),
        [`${baseUrl}__classification/inputs/${item.id}.f32`, item.inputSha256]);
      freshTrace.images.push({ id: item.id, logits }); await writeJson(resolve(output, 'fresh-browser.json'), freshTrace);
    }
    await freshPage.evaluate(async () => window.realVision.close()); await fresh.close();
    status.phase = 'independent-python'; await save();
    const pythonArgs = [resolve(root, 'scripts/classification_real_reference.py'), '--plan', planPath, '--data', dataPath,
      '--images', resolve(args.images), '--run', output, '--source-model', sourceModel, '--source-cache', resolve(args.cache)];
    try {
      const result = await exec(args.python, pythonArgs, { cwd: root, timeout: remainingTimeoutMs(deadline, performance.now()), maxBuffer: 1024 * 1024 });
      await writeFile(resolve(output, 'python.log'), `${result.stdout}\n${result.stderr}`);
    } catch (error) {
      await writeFile(resolve(output, 'python.log'), `${error.stdout ?? ''}\n${error.stderr ?? ''}\n${String(error)}`);
      throw new Error(`Independent Python comparison failed: ${error.stderr ?? error.message}`);
    }
    const comparison = await json(resolve(output, 'comparison.json'));
    if (comparison.status !== 'completed' || typeof comparison.passed !== 'boolean') throw new Error('Incomplete Python comparison');
    if (unexpectedRequests.length || pageErrors.length) throw new Error('Unexpected browser traffic or page error');
    status.status = 'completed'; status.phase = 'completed'; status.passed = comparison.passed;
    status.modelSha256 = modelSha256; status.comparisonSha256 = sha(await readFile(resolve(output, 'comparison.json')));
    await save(); console.log(JSON.stringify(status));
  } catch (error) {
    if (active) { status.status = 'error'; status.passed = false; status.error = String(error); await save(); }
    throw error;
  } finally {
    clearTimeout(timer);
    if (active) await writeJson(resolve(output, 'network.json'), { requests, unexpectedRequests, pageErrors });
    await browser?.close(); await new Promise(done => server ? server.close(done) : done());
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
