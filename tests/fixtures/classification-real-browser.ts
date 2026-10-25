// Developer-only measurement helpers. The actual training remains in App/Worker.
import * as ort from 'onnxruntime-web/wasm';
import { decodeImage } from '../../src/ml/decode';
import { preprocessRgba } from '../../src/ml/preprocess';
import { predictHead } from '../../src/ml/head';
import type { HeadWeights } from '../../src/ml/types';

let session: ort.InferenceSession | undefined;
let head: HeadWeights | undefined;

async function sha(bytes: Uint8Array): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer)), b => b.toString(16).padStart(2, '0')).join('');
}

async function checkedBytes(url: string, expected: string): Promise<Uint8Array> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Fixture GET failed: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (await sha(bytes) !== expected) throw new Error('Fixture byte hash mismatch');
  return bytes;
}

async function initialize(url: string, expectedHash: string, runtime: string): Promise<void> {
  if (session) throw new Error('Measurement session already initialized');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = runtime;
  session = await ort.InferenceSession.create(await checkedBytes(url, expectedHash), {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all',
  });
}

export async function initializeSource(url: string, expectedHash: string, runtime: string,
  values: { weights: number[]; bias: number[]; classes: number; featureSize: number }): Promise<void> {
  head = { ...values, weights: new Float32Array(values.weights), bias: new Float32Array(values.bias) };
  await initialize(url, expectedHash, runtime);
}

export async function initializeExport(url: string, expectedHash: string, runtime: string): Promise<void> {
  if (head) throw new Error('Fresh inference must not inherit a training head');
  await initialize(url, expectedHash, runtime);
}

async function infer(input: Float32Array): Promise<Record<string, number[]>> {
  if (!session || input.length !== 3 * 224 * 224 || !input.every(Number.isFinite)) throw new Error('Invalid measurement input');
  const tensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
  let outputs: ort.InferenceSession.OnnxValueMapType | undefined;
  try {
    outputs = await session.run({ images: tensor });
    return Object.fromEntries(Object.entries(outputs).map(([name, value]) => {
      if (!(value.data instanceof Float32Array) || !value.data.every(Number.isFinite)) throw new Error('Nonfinite/non-FP32 output');
      return [name, Array.from(value.data)];
    }));
  } finally {
    tensor.dispose();
    if (outputs) Object.values(outputs).forEach(value => value.dispose());
  }
}

function base64(bytes: Uint8Array): string {
  let text = '';
  for (let offset = 0; offset < bytes.length; offset += 32768) text += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  return btoa(text);
}

export async function collectImage(url: string, expectedHash: string, expectedRgbaHash: string) {
  if (!head) throw new Error('Missing extracted final head');
  const encoded = await checkedBytes(url, expectedHash);
  const { rgba, width, height } = await decodeImage(new Blob([new Uint8Array(encoded).buffer], { type: 'image/png' }));
  if (await sha(new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength)) !== expectedRgbaHash) throw new Error('Decoded canonical pixels differ');
  const input = preprocessRgba(rgba, width, height);
  const inputBytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  const values = await infer(input);
  if (values.features?.length !== 1280 || values.logits?.length !== 1000) throw new Error('Unexpected source output shapes');
  return { inputSha256: await sha(inputBytes), inputBase64: base64(inputBytes), features: values.features,
    baseLogits: values.logits, finalLogits: Array.from(predictHead(new Float32Array(values.features), head)) };
}

export async function inferExport(url: string, expectedHash: string) {
  const bytes = await checkedBytes(url, expectedHash);
  if (bytes.length !== 3 * 224 * 224 * 4) throw new Error('Invalid input byte length');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const input = Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
  const values = await infer(input);
  if (Object.keys(values).length !== 1 || values.logits?.length !== 5) throw new Error('Unexpected standalone output');
  return values.logits;
}

export async function close(): Promise<void> {
  await session?.release(); session = undefined; head = undefined;
}
