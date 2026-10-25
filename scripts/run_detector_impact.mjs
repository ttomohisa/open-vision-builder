// Local developer controller. The browser receives public fixtures via GET only.
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { createServer } from 'node:http';
import { lstat, mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs, promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { chromium } from '@playwright/test';
import { ImpactProgress, validateImpactPlan, remainingTimeoutMs } from './detector-impact-state.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const { values: args } = parseArgs({ options: {
  check: { type: 'boolean', default: false },
  python: { type: 'string', default: resolve(root, 'artifacts/p1b/.venv-detector/bin/python') },
  assets: { type: 'string', default: resolve(root, 'artifacts/p1b/detector-assets/unqualified') },
  cache: { type: 'string', default: resolve(root, 'artifacts/p1b/detector-assets/source') },
  data: { type: 'string', default: resolve(root, 'artifacts/p1d/data') },
  output: { type: 'string', default: resolve(root, 'artifacts/p1d/run') },
  chromium: { type: 'string' },
} });
const planPath = resolve(root, 'tests/fixtures/detector-impact-plan.json');
const manifestPath = resolve(root, 'tests/fixtures/detector-impact-data.json');
const sha = value => createHash('sha256').update(value).digest('hex');
const exec = promisify(execFile);
async function json(path) { return JSON.parse(await readFile(path, 'utf8')); }
async function atomicJson(path, value) {
  const temporary = `${path}.part`;
  for (const candidate of [path, temporary]) {
    const info = await lstat(candidate).catch(error => error.code === 'ENOENT' ? null : Promise.reject(error));
    if (info?.isSymbolicLink()) throw new Error('Refusing a symlink report');
  }
  await mkdir(dirname(path), { recursive: true });
  await writeFile(temporary, JSON.stringify(value) + '\n', { flag: 'w' });
  await rename(temporary, path);
}
async function freshDirectory(path) {
  // Never replace an earlier experiment or unrelated file. Parent symlinks are disallowed.
  for (let cursor = resolve(path); ; cursor = dirname(cursor)) {
    const info = await lstat(cursor).catch(error => error.code === 'ENOENT' ? null : Promise.reject(error));
    if (info?.isSymbolicLink()) throw new Error('Refusing a symlink experiment directory');
    if (dirname(cursor) === cursor) break;
  }
  const entries = await readdir(path).catch(error => error.code === 'ENOENT' ? [] : Promise.reject(error));
  if (entries.length) throw new Error('Experiment output already exists; keep the original result. No automatic rerun.');
  await mkdir(path, { recursive: true });
}

async function main() {
  const [planBytes, manifestBytes] = await Promise.all([readFile(planPath), readFile(manifestPath)]);
  const plan = JSON.parse(planBytes), manifest = JSON.parse(manifestBytes);
  const planSha256 = sha(planBytes), datasetManifestSha256 = sha(manifestBytes);
  validateImpactPlan(plan, datasetManifestSha256);
  const train = manifest.images.filter(item => item.split === 'train');
  const progress = new ImpactProgress(train.map(item => item.id));
  const output = resolve(args.output), referenceDir = resolve(output, 'reference');
  const statusPath = resolve(output, 'status.json');
  const tracePath = resolve(output, 'browser.json');
  const requests = [], unexpectedRequests = [], pageErrors = [], controls = [];
  const trace = {
    schemaVersion: 1, planSha256, datasetManifestSha256, status: 'running',
    images: manifest.images.map(item => ({ id: item.id, split: item.split, targets: item.targets, inputSha256: item.input.sha256 })),
    initial: [], final: [], steps: [], checkpoints: {}, frozenUnchanged: true,
    detectorQualification: 'FAIL', releaseReady: false,
  };
  const status = { schemaVersion: 1, planSha256, datasetManifestSha256, status: 'running',
    passed: false, detectorQualification: 'FAIL', releaseReady: false, controls };
  let server, browser, timer, runtimeStart;
  let active = false;
  async function save() {
    status.progress = progress.snapshot();
    status.durationSeconds = runtimeStart ? (performance.now() - runtimeStart) / 1000 : 0;
    await atomicJson(tracePath, trace);
    await atomicJson(statusPath, status);
  }
  async function python(script, extra, expectedOutput) {
    const common = ['--data-dir', args.data, '--manifest', manifestPath, '--plan', planPath];
    let exitCode = 0, stdout = '', stderr = '';
    try {
      ({ stdout, stderr } = await exec(args.python, [resolve(root, 'scripts', script), ...common, ...extra],
        { cwd: root, timeout: remainingTimeoutMs(runtimeStart + plan.budget.runtimeMinutes * 60_000, performance.now()), maxBuffer: 1024 * 1024 }));
    } catch (error) {
      if (typeof error.code !== 'number') throw error;
      exitCode = error.code; stdout = error.stdout ?? ''; stderr = error.stderr ?? '';
    }
    await writeFile(`${expectedOutput}.log`, `${stdout}\n${stderr}`);
    const result = await json(expectedOutput);
    if (result.status === 'error' || result.error || (exitCode !== 0 && result.passed !== false)) {
      throw new Error(`Python ${script} failed: ${result.error ?? stderr ?? exitCode}`);
    }
    return result;
  }
  try {
    const resources = new Map();
    const add = (path, body, type = 'application/octet-stream') => resources.set(path, { body, type });
    add('plan.json', planBytes, 'application/json'); add('data.json', manifestBytes, 'application/json');
    const prepared = await json(resolve(args.assets, 'model.json'));
    add('model/model.json', await readFile(resolve(args.assets, 'model.json')), 'application/json');
    for (const spec of Object.values(prepared.artifacts)) {
      const value = await readFile(resolve(args.assets, spec.file));
      if (value.length !== spec.bytes || sha(value) !== spec.sha256) throw new Error('Prepared artifact digest mismatch');
      add(`model/${spec.file}`, value);
    }
    for (const item of manifest.images) {
      if (item.input.file !== `inputs/${item.id}.f32`) throw new Error('Invalid input path');
      const value = await readFile(resolve(args.data, item.input.file));
      if (value.length !== item.input.bytes || sha(value) !== item.input.sha256) throw new Error('Canonical input digest mismatch');
      add(`data/${item.input.file}`, value);
    }
    for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
      add(`runtime/${name}`, await readFile(resolve(root, 'node_modules/onnxruntime-web/dist', name)),
        name.endsWith('.mjs') ? 'text/javascript' : 'application/wasm');
    }
    const built = await build({ configFile: false, logLevel: 'silent', build: {
      write: false, target: 'es2022', minify: false, assetsInlineLimit: 0,
      rolldownOptions: { input: resolve(root, 'tests/fixtures/detection-impact-browser.ts'), preserveEntrySignatures: 'strict',
        output: { codeSplitting: false, entryFileNames: 'harness.js' } },
    } });
    const outputs = (Array.isArray(built) ? built : [built]).flatMap(item => item.output ?? []);
    const entry = outputs.find(item => item.type === 'chunk' && item.isEntry);
    if (!entry || entry.imports.length || entry.dynamicImports.some(name => name !== entry.fileName)) throw new Error('Unexpected external harness dependency');
    for (const item of outputs) add(item.fileName, item.type === 'chunk' ? item.code : item.source,
      item.fileName.endsWith('.js') ? 'text/javascript' : item.fileName.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    add('', '<!doctype html><title>Local D-FINE impact experiment</title><link rel="icon" href="data:,">', 'text/html');
    const prefix = '/open-vision-builder/__impact/';
    server = createServer((request, response) => {
      const path = request.url?.split('?')[0] ?? '';
      const key = path.startsWith(prefix) ? path.slice(prefix.length) : null;
      const value = key === null ? null : resources.get(key);
      if (request.method !== 'GET' || !value) {
        unexpectedRequests.push(`${request.method} ${path}`);
        response.writeHead(404).end(); return;
      }
      response.writeHead(200, { 'Content-Type': value.type, 'Cache-Control': 'no-store' }).end(value.body);
    });
    await new Promise((done, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', done); });
    const baseUrl = `http://127.0.0.1:${server.address().port}${prefix}`;
    const moduleUrl = new URL('harness.js', baseUrl).href;
    browser = await chromium.launch({ headless: true, ...(args.chromium ? { executablePath: args.chromium } : {}) });
    async function newPage(fresh = false) {
      const context = await browser.newContext({ acceptDownloads: true });
      await context.route('**/*', async route => {
        const request = route.request(), url = request.url();
        if (request.method() !== 'GET' || !url.startsWith(baseUrl) || (fresh && /\/(?:model|plan\.json|data\.json)(?:\/|$)/.test(url))) {
          unexpectedRequests.push(`${request.method()} ${url}`); await route.abort(); return;
        }
        requests.push({ phase: fresh ? 'fresh' : 'training', method: request.method(), url: url.replace(baseUrl, './') });
        await route.continue();
      });
      const page = await context.newPage();
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.goto(baseUrl);
      await page.evaluate(async url => { globalThis.impact = await import(url); }, moduleUrl);
      return { context, page };
    }
    let { context, page } = await newPage();
    if (args.check) {
      console.log(JSON.stringify({ status: 'preflight-passed', browser: browser.version(), planSha256, datasetManifestSha256,
        note: 'Bundling, fixed bytes, and browser module import only. No model inference or training.' }));
      return;
    }
    await freshDirectory(output); active = true; runtimeStart = performance.now();
    status.startedAt = new Date().toISOString();
    status.environment = { node: process.version, platform: process.platform, architecture: process.arch, browser: browser.version() };
    status.harnessSha256 = sha(entry.code);
    status.codeSha256 = {};
    for (const file of ['scripts/run_detector_impact.mjs', 'scripts/detector-impact-state.mjs', 'scripts/detector_impact_reference.py',
      'scripts/verify_detector_impact_export.py', 'tests/fixtures/detection-impact-browser.ts', 'tests/fixtures/detector-training-tolerances.json']) {
      status.codeSha256[file] = sha(await readFile(resolve(root, file)));
    }
    await save();
    timer = setTimeout(() => browser.close().catch(() => {}), plan.budget.runtimeMinutes * 60_000);
    const initialized = await page.evaluate(async values => globalThis.impact.initialize(...values), [baseUrl, planSha256, datasetManifestSha256]);
    Object.assign(trace, { environment: initialized.environment, prefixSha256: initialized.prefixSha256, sourceModelSha256: initialized.sourceModelSha256 });
    trace.checkpoints['0'] = initialized.weights;
    async function control(checkpoint) {
      console.log(`Same-feature control at step ${checkpoint}`);
      const measured = await page.evaluate(id => globalThis.impact.measure(id, true), train[0].id);
      const measurementPath = resolve(output, `control-${checkpoint}-browser.json`), resultPath = resolve(output, `control-${checkpoint}.json`);
      await atomicJson(measurementPath, measured);
      const result = await python('detector_impact_reference.py', ['--mode', 'control', '--assets', args.assets,
        '--browser', measurementPath, '--output', resultPath], resultPath);
      controls.push({ step: checkpoint, file: `control-${checkpoint}.json`, passed: result.passed, sha256: sha(await readFile(resultPath)) });
      progress.recordControl(checkpoint, result.passed);
      trace.checkpoints[String(checkpoint)] = measured.weights;
      await save();
      if (!result.passed) {
        const error = new Error(`Same-feature control failed at step ${checkpoint}`); error.controlFailure = true; throw error;
      }
    }
    await control(0);
    console.log('Generating the fixed PyTorch own-prefix trajectory');
    await python('detector_impact_reference.py', ['--mode', 'reference', '--assets', args.assets, '--cache', args.cache,
      '--output-dir', referenceDir], resolve(referenceDir, 'summary.json'));
    async function evaluate(stage) {
      for (const item of manifest.images) {
        console.log(`Browser ${stage}: ${item.id}`);
        const measured = await page.evaluate(id => globalThis.impact.measure(id, false), item.id);
        trace[stage].push({ id: item.id, prediction: measured.prediction, losses: measured.losses, matches: measured.matches });
        if (stage === 'initial') await atomicJson(resolve(output, 'features', `${item.id}.json`), measured.features);
        trace.frozenUnchanged &&= measured.frozenUnchanged;
        await save();
      }
    }
    await evaluate('initial');
    for (let index = 0; index < plan.recipe.steps; index++) {
      if (index === 6) await control(6);
      const id = train[index % train.length].id;
      console.log(`Browser update ${index + 1}/${plan.recipe.steps}: ${id}`);
      const measured = await page.evaluate(id => globalThis.impact.measure(id, true), id);
      // Persist the proposal before accepting it, so an interruption preserves the actual tensors.
      await atomicJson(resolve(output, 'steps', `${index + 1}.json`), measured);
      const accepted = await page.evaluate(({ id, index }) => globalThis.impact.acceptUpdate(id, index), { id, index });
      if (accepted !== index + 1) throw new Error('Browser step acknowledgment mismatch');
      progress.acceptStep(index, id);
      const { features, weights, config, ...stepTrace } = measured;
      trace.steps.push({ ...stepTrace, step: accepted });
      trace.frozenUnchanged &&= measured.frozenUnchanged;
      await save();
    }
    await control(12);
    await evaluate('final');
    const download = page.waitForEvent('download', { timeout: 30_000 });
    trace.artifact = await page.evaluate(() => globalThis.impact.exportModel());
    await (await download).saveAs(resolve(output, 'model.onnx'));
    const modelBytes = await readFile(resolve(output, 'model.onnx'));
    if (sha(modelBytes) !== trace.artifact.sha256 || modelBytes.length !== trace.artifact.bytes) throw new Error('Downloaded model differs');
    add('export/model.onnx', modelBytes);
    await context.close();
    ({ context, page } = await newPage(true));
    const fresh = await page.evaluate(values => globalThis.impact.verifyExport(...values),
      [baseUrl, { file: 'export/model.onnx', ...trace.artifact }, manifest.images.map(({ id, input }) => ({ id, input }))]);
    await atomicJson(resolve(output, 'fresh-browser.json'), fresh);
    trace.status = 'completed'; await save();
    const exportPath = resolve(output, 'export-verification.json');
    const exported = await python('verify_detector_impact_export.py', ['--model', resolve(output, 'model.onnx'),
      '--source', resolve(args.assets, 'detector.onnx'), '--browser', tracePath,
      '--fresh-browser', resolve(output, 'fresh-browser.json'), '--recipe', resolve(args.assets, 'model.json'), '--output', exportPath], exportPath);
    const comparisonPath = resolve(output, 'comparison.json');
    const comparison = await python('detector_impact_reference.py', ['--mode', 'finalize', '--browser', tracePath,
      '--output-dir', referenceDir, '--output', comparisonPath], comparisonPath);
    if (unexpectedRequests.length || pageErrors.length) throw new Error('Unexpected browser communication or page errors');
    progress.finish(); status.status = 'completed';
    status.passed = comparison.passed && exported.passed;
    status.decision = !exported.passed ? 'export-parity-outside-existing-conditions' : comparison.decision;
    status.comparison = { file: 'comparison.json', sha256: sha(await readFile(comparisonPath)) };
    status.export = { file: 'export-verification.json', passed: exported.passed, sha256: sha(await readFile(exportPath)) };
    await save();
    console.log(JSON.stringify({ status: status.status, passed: status.passed, decision: status.decision, output }));
  } catch (error) {
    if (!active) throw error;
    status.status = error.controlFailure ? 'stopped' : 'error'; status.reason = String(error);
    status.decision = error.controlFailure ? 'investigate-head-control-before-own-path-training' : 'inconclusive-execution-error';
    trace.status = status.status; progress.stop(status.reason); await save();
    console.error(JSON.stringify({ status: status.status, passed: false, reason: status.reason, output }));
    // An observed numerical stop is recorded evidence. Infrastructure/integrity errors fail the command.
    if (!error.controlFailure) process.exitCode = 1;
  } finally {
    clearTimeout(timer);
    if (active) await atomicJson(resolve(output, 'network.json'), { requests, unexpectedRequests, pageErrors });
    await browser?.close();
    if (server) await new Promise(done => server.close(done));
  }
}
await main();
