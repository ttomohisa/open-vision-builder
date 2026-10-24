/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/wasm';
import { zipSync, strToU8 } from 'fflate';
import { trainHead, predictHead } from './ml/head';
import { exportClassifier } from './ml/onnx';
import { preprocessRgba, referenceRgba } from './ml/preprocess';
import { decodeImage } from './ml/decode';
import type { ModelManifest, NativeReference } from './ml/types';
import type { RunRequest, RunSummary, WorkerRequest, WorkerResponse, Prediction } from './protocol';

const worker = self as unknown as DedicatedWorkerGlobalScope;
const base = new URL(import.meta.env.BASE_URL, worker.location.origin);
const modelRoot = new URL('models/mobilenetv4-small/', base);
const ATOL = 2e-4, RTOL = 1e-4;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL('runtime/onnxruntime/', base).href;

class Cancelled extends Error {}
let active: { id: number; cancelled: boolean } | null = null;
type Prepared = { manifest: ModelManifest; bytes: Uint8Array; session: ort.InferenceSession; reference: NativeReference; nativeError: number; referenceInput: Float32Array; referenceFeatures: Float32Array };
let prepared: Promise<Prepared> | null = null;

const post = (message: WorkerResponse, transfers: Transferable[] = []) => worker.postMessage(message, transfers);
const check = (id: number) => { if (!active || active.id !== id || active.cancelled) throw new Cancelled(); };
const progress = (id: number, stage: string, fraction: number, loss?: number) => post({ type: 'progress', id, stage, fraction, loss });

async function get(url: URL): Promise<Response> {
  if (url.origin !== base.origin) throw new Error('静的アセットは同じサイトからのみ読み込めます。');
  const response = await fetch(url, { credentials: 'omit' });
  if (!response.ok) throw new Error(`必要なファイルを取得できませんでした: ${url.pathname} (${response.status})`);
  return response;
}

function asset(name: string): URL {
  if (!/^[A-Za-z0-9._-]+$/.test(name)) throw new Error('不正なモデルファイル名です。');
  return new URL(name, modelRoot);
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer);
  return Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
}

function compare(actual: ArrayLike<number>, expected: ArrayLike<number>, context: string): number {
  if (actual.length !== expected.length || !actual.length) throw new Error(`${context}: 出力の形状が一致しません。`);
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i]) || error > ATOL + RTOL * Math.abs(expected[i])) {
      throw new Error(`${context}: 数値検証に失敗しました (index ${i}, error ${error})。`);
    }
    maxAbs = Math.max(maxAbs, error);
  }
  return maxAbs;
}

async function infer(session: ort.InferenceSession, name: string, input: Float32Array): Promise<Record<string, Float32Array>> {
  const tensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
  let outputs: ort.InferenceSession.OnnxValueMapType | undefined;
  try {
    outputs = await session.run({ [name]: tensor });
    const copies: Record<string, Float32Array> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (!(value.data instanceof Float32Array)) throw new Error('FP32 以外のモデル出力です。');
      copies[key] = new Float32Array(value.data);
    }
    return copies;
  } finally {
    tensor.dispose();
    if (outputs) Object.values(outputs).forEach(value => value.dispose());
  }
}

async function loadModel(): Promise<Prepared> {
  const manifest = await (await get(asset('model.json'))).json() as ModelManifest;
  if (manifest.schemaVersion !== 1 || manifest.id !== 'mobilenetv4-small' || manifest.features.size !== 1280 ||
      manifest.input.resize !== 'bilinear-half-pixel-stretch' || JSON.stringify(manifest.input.shape) !== '[1,3,224,224]' ||
      JSON.stringify(manifest.input.mean) !== '[0.485,0.456,0.406]' || JSON.stringify(manifest.input.std) !== '[0.229,0.224,0.225]' ||
      manifest.source.sha256 !== '7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4') throw new Error('この版で検証したモデル仕様と一致しません。');
  const url = asset(manifest.artifact.file);
  url.searchParams.set('sha256', manifest.artifact.sha256);
  const bytes = new Uint8Array(await (await get(url)).arrayBuffer());
  if (bytes.length !== manifest.artifact.bytes || await sha256(bytes) !== manifest.artifact.sha256) throw new Error('モデルの整合性確認に失敗しました。再読み込みしてください。');
  const reference = await (await get(asset(manifest.reference.file))).json() as NativeReference;
  if (reference.schemaVersion !== 1 || reference.fixture.name !== 'rgba-pattern-v1' || reference.fixture.width !== 40 || reference.fixture.height !== 32) throw new Error('未知の参照データです。');
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  try {
    const referenceInput = preprocessRgba(referenceRgba(), 40, 32);
    const outputs = await infer(session, manifest.input.name, referenceInput);
    const nativeError = Math.max(compare(outputs[manifest.features.name], reference.native.features, '元モデルの特徴'), compare(outputs[manifest.originalOutput], reference.native.logits, '元モデルの分類'));
    return { manifest, bytes, reference, session, nativeError, referenceInput, referenceFeatures: outputs[manifest.features.name] };
  } catch (error) { await session.release(); throw error; }
}

function validate(request: RunRequest): string[] {
  const names = request.labels.map(x => x.trim().normalize('NFC'));
  if (names.length < 2 || names.length > 8 || new Set(names).size !== names.length || names.some(n => !n || n.length > 80)) throw new Error('異なる名前のクラスを2〜8個用意してください。');
  if (request.samples.length > 64 || !request.samples.length || new Set(request.samples.map(s => s.id)).size !== request.samples.length) throw new Error('画像は合計64枚以内で指定してください。');
  if (!Number.isInteger(request.epochs) || request.epochs < 1 || request.epochs > 200 || !Number.isFinite(request.learningRate) || request.learningRate < 0.001 || request.learningRate > 0.1) throw new Error('学習設定が範囲外です。');
  const counts = names.map(() => ({ train: 0, test: 0 }));
  for (const sample of request.samples) {
    if (!Number.isInteger(sample.label) || !counts[sample.label] || !['train', 'test'].includes(sample.split) ||
        !['image/png', 'image/jpeg', 'image/webp'].includes(sample.file.type) || sample.file.size > 20 * 1024 * 1024) throw new Error('PNG・JPEG・WebPの画像と正しいクラス・分割を指定してください。');
    counts[sample.label][sample.split]++;
  }
  if (counts.some(c => c.train < 1 || c.test < 1)) throw new Error('各クラスに学習用と評価用の画像を1枚以上用意してください。');
  return names;
}

async function imageTensor(file: Blob): Promise<Float32Array> {
  const { rgba, width, height } = await decodeImage(file);
  return preprocessRgba(rgba, width, height);
}

function probabilities(logits: Float32Array): number[] {
  const max = Math.max(...logits);
  const exp = Array.from(logits, x => Math.exp(x - max));
  const total = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / total);
}

async function run(request: RunRequest): Promise<void> {
  const { id } = request;
  const started = performance.now();
  const labels = validate(request);
  progress(id, 'モデルを読み込み、元モデルとの一致を確認しています', 0.04);
  if (!prepared) prepared = loadModel().catch(error => { prepared = null; throw error; });
  const model = await prepared;
  check(id);
  const trainFeatures: number[] = [], trainLabels: number[] = [];
  const tests: { input: Float32Array; features: Float32Array; id: string; actual: number }[] = [];
  const seenImages = new Set<string>();
  for (let i = 0; i < request.samples.length; i++) {
    check(id);
    const sample = request.samples[i];
    progress(id, `画像の特徴を抽出しています (${i + 1}/${request.samples.length})`, 0.15 + 0.35 * i / request.samples.length);
    const input = await imageTensor(sample.file);
    const imageHash = await sha256(new Uint8Array(input.buffer));
    if (seenImages.has(imageHash)) throw new Error('同一の画像が重複しています。学習用と評価用には別の画像を使ってください。');
    seenImages.add(imageHash);
    const outputs = await infer(model.session, model.manifest.input.name, input);
    const features = outputs[model.manifest.features.name];
    if (sample.split === 'train') { trainFeatures.push(...features); trainLabels.push(sample.label); }
    else tests.push({ input, features, id: sample.id, actual: sample.label });
  }
  check(id);
  const trained = await trainHead(new Float32Array(trainFeatures), new Int32Array(trainLabels), {
    classes: labels.length, featureSize: model.manifest.features.size, epochs: request.epochs, learningRate: request.learningRate, backend: 'wasm',
    isCancelled: () => !active || active.id !== id || active.cancelled,
    onEpoch: (epoch, loss) => progress(id, `分類器を学習しています (${epoch}/${request.epochs})`, 0.5 + 0.3 * epoch / request.epochs, loss),
  });
  check(id);
  if (trained.cancelled) throw new Cancelled();
  const weightDeltaL2 = Math.sqrt([...trained.head.weights, ...trained.head.bias].reduce((sum, value) => sum + value * value, 0));
  if (!(weightDeltaL2 > 0) || !Number.isFinite(weightDeltaL2)) throw new Error('重みの更新を確認できませんでした。クラスごとに異なる画像を用意してください。');
  progress(id, '完全なONNXを生成し、別の実行セッションで検証しています', 0.84);
  const bytes = exportClassifier(model.bytes, model.manifest, trained.head, labels);
  const fresh = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const expectedFixture = predictHead(model.referenceFeatures, trained.head);
  let exportError = 0;
  const predictions: Prediction[] = [];
  try {
    const referenceOutput = await infer(fresh, model.manifest.input.name, model.referenceInput);
    exportError = compare(referenceOutput.logits, expectedFixture, '書き出したモデルの参照出力');
    for (const sample of tests) {
      check(id);
      const expected = predictHead(sample.features, trained.head);
      const output = await infer(fresh, model.manifest.input.name, sample.input);
      exportError = Math.max(exportError, compare(output.logits, expected, '評価画像の書き出し結果'));
      const scores = probabilities(output.logits);
      predictions.push({ id: sample.id, actual: sample.actual, predicted: scores.indexOf(Math.max(...scores)), scores });
    }
  } finally { await fresh.release(); }
  check(id);
  const accuracy = predictions.filter(p => p.actual === p.predicted).length / predictions.length;
  const modelHash = await sha256(bytes);
  const summary: RunSummary = { epochs: trained.epochsCompleted, losses: trained.losses, accuracy, predictions, trainingCount: trainLabels.length, testCount: tests.length, durationMs: performance.now() - started, nativeMaxAbsError: model.nativeError, exportMaxAbsError: exportError, weightDeltaL2, modelBytes: bytes.length, modelSha256: modelHash, backend: trained.backend };
  const report = {
    schemaVersion: 1, labels, model: model.manifest.source,
    training: { scope: 'classifier-head-only', epochsCompleted: trained.epochsCompleted, learningRate: request.learningRate, backend: trained.backend, engine: '@jax-js/jax@0.1.27', losses: trained.losses, lossTiming: 'before-each-update', weightDeltaL2 },
    metrics: { accuracy, trainingCount: trainLabels.length, testCount: tests.length, durationMs: summary.durationMs },
    verification: { atol: ATOL, rtol: RTOL, native: { passed: true, maxAbsError: model.nativeError }, export: { passed: true, maxAbsError: exportError, images: tests.length + 1 } },
    artifact: { file: 'model.onnx', sha256: modelHash, bytes: bytes.length },
    environment: { userAgent: navigator.userAgent, runtime: 'onnxruntime-web@1.30.0', threads: 1, crossOriginIsolated, peakMemory: 'NOT MEASURED' },
    parityFixture: { source: 'generated-public-rgba-pattern-v1', shape: [1, 3, 224, 224], input: Array.from(model.referenceInput), expectedLogits: Array.from(expectedFixture) },
    limitations: ['Frozen backbone; only the classification head is trained.', 'No detection or WebGPU qualification.', 'Synthetic samples demonstrate the pipeline, not real-world accuracy.'],
  };
  const reportText = JSON.stringify(report, null, 2);
  const deployment = { schemaVersion: 1, task: 'image-classification', model: { file: 'model.onnx', sha256: modelHash }, input: { ...model.manifest.input, color: 'RGB', alpha: 'composite-white-float' }, output: { name: 'logits', shape: [1, labels.length], postprocess: 'softmax', labels: 'labels.json' }, source: model.manifest.source, recipe: 'mobilenetv4-small-frozen-backbone-linear-head-v1' };
  progress(id, 'モデル・実行例・ライセンスをまとめています', 0.97);
  const licenseFiles = [model.manifest.licenses.licenseFile, model.manifest.licenses.noticeFile, model.manifest.licenses.modelCardFile];
  const texts = await Promise.all(licenseFiles.map(async name => (await get(asset(name))).text()));
  const kit = await Promise.all(['predict.py', 'README.txt'].map(async name => (await get(new URL(`kit/${name}`, base))).text()));
  check(id);
  const archive = zipSync({
    'model.onnx': bytes, 'labels.json': strToU8(JSON.stringify(labels, null, 2)), 'manifest.json': strToU8(JSON.stringify(deployment, null, 2)), 'report.json': strToU8(reportText),
    'LICENSE.txt': strToU8(texts[0]), 'NOTICE.txt': strToU8(`${texts[1]}\nFurther modification: Open Vision Builder replaced the original classifier with a locally trained ${labels.length}-class head. The complete frozen feature extractor is included.\n`),
    'MODEL_CARD.md': strToU8(texts[2]), 'predict.py': strToU8(kit[0]), 'README.txt': strToU8(kit[1]),
  }, { level: 1 });
  check(id);
  const archiveBuffer = new Uint8Array(archive).buffer;
  const modelBuffer = new Uint8Array(bytes).buffer;
  post({ type: 'result', id, summary, archive: archiveBuffer, model: modelBuffer, report: reportText }, [archiveBuffer, modelBuffer]);
}

worker.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === 'cancel') { if (active?.id === request.id) active.cancelled = true; return; }
  if (active) { post({ type: 'error', id: request.id, message: '先に実行中の学習を中断してください。' }); return; }
  active = { id: request.id, cancelled: false };
  void run(request).catch(error => {
    if (error instanceof Cancelled) post({ type: 'cancelled', id: request.id });
    else post({ type: 'error', id: request.id, message: error instanceof Error ? error.message : String(error) });
  }).finally(() => { if (active?.id === request.id) active = null; });
};
