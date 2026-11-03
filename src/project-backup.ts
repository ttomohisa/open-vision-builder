import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import type { Sample } from './protocol';

// Project backups are separate from inference-only model ZIPs and training checkpoints.
export const MAX_PROJECT_ARCHIVE_BYTES = 512 * 1024 * 1024;
const MAX_IMAGES = 64;
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_PIXELS = 16_000_000;
const MAX_MANIFEST_BYTES = 64 * 1024;
const IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const encoder = new TextEncoder();

export interface ClassificationProject {
  labels: string[];
  samples: Sample[];
  epochs: number;
  learningRate: number;
  syntheticIds?: string[];
}

interface ImageEntry {
  id: string; name: string; label: number; split: 'train' | 'test';
  path: string; type: string; size: number; sha256: string; synthetic: boolean;
}
interface Manifest {
  kind: 'open-vision-classification-project';
  schemaVersion: 1;
  recipe: 'mobilenetv4-small-frozen-head-v1';
  labels: string[];
  epochs: number;
  learningRate: number;
  images: ImageEntry[];
}

function fail(message: string): never {
  throw new Error(`プロジェクトZIPを処理できません: ${message}`);
}
function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function validLabels(labels: unknown): labels is string[] {
  return Array.isArray(labels) && labels.length >= 2 && labels.length <= 8
    && labels.every(label => typeof label === 'string' && label.trim() === label
      && label.length > 0 && label.length <= 80 && label.normalize('NFC') === label)
    && new Set(labels).size === labels.length;
}
function validSettings(epochs: unknown, learningRate: unknown): boolean {
  return Number.isInteger(epochs) && (epochs as number) >= 1 && (epochs as number) <= 200
    && typeof learningRate === 'number' && Number.isFinite(learningRate)
    && learningRate >= 0.001 && learningRate <= 0.1;
}
function validId(id: unknown): id is string {
  return typeof id === 'string' && id.length >= 1 && id.length <= 128
    && !/[\x00-\x1f\x7f]/.test(id);
}
function pathFor(index: number): string {
  return `images/${String(index).padStart(3, '0')}.bin`;
}
async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
function validateImageName(name: unknown): name is string {
  return typeof name === 'string' && name.length > 0 && name.length <= 255
    && !/[\x00-\x1f\x7f]/.test(name);
}
function validateSampleMeta(sample: Sample, labels: string[], seen: Set<string>) {
  if (!validId(sample.id) || seen.has(sample.id)) fail('画像IDが重複または不正です。');
  seen.add(sample.id);
  if (!validateImageName(sample.name) || !Number.isInteger(sample.label)
      || sample.label < 0 || sample.label >= labels.length
      || (sample.split !== 'train' && sample.split !== 'test')
      || !IMAGE_TYPES.has(sample.file.type) || sample.file.size <= 0
      || sample.file.size > MAX_FILE_BYTES) fail('画像の名前・ラベル・形式・サイズが不正です。');
}
function validateProject(project: ClassificationProject) {
  if (!validLabels(project.labels)) fail('クラス名は重複なしの2〜8個、各80文字以内にしてください。');
  if (!validSettings(project.epochs, project.learningRate)) fail('学習設定が不正です。');
  if (!Array.isArray(project.samples) || project.samples.length > MAX_IMAGES) fail('画像が64枚を超えています。');
  const seen = new Set<string>();
  for (const sample of project.samples) validateSampleMeta(sample, project.labels, seen);
}

export async function exportClassificationProject(project: ClassificationProject): Promise<Uint8Array> {
  validateProject(project);
  const entries: Record<string, Uint8Array> = {};
  const images: ImageEntry[] = [];
  let total = 0;
  const synthetic = new Set(project.syntheticIds ?? []);
  for (const [index, sample] of project.samples.entries()) {
    const bytes = new Uint8Array(await sample.file.arrayBuffer());
    if (bytes.byteLength !== sample.file.size) fail('保存中に画像サイズが変わりました。');
    total += bytes.byteLength;
    if (total > MAX_PROJECT_ARCHIVE_BYTES - MAX_MANIFEST_BYTES - 16384) {
      fail('バックアップの上限512MiBを超えます。画像を減らしてください。');
    }
    const path = pathFor(index);
    entries[path] = bytes;
    images.push({
      id: sample.id, name: sample.name, label: sample.label, split: sample.split,
      type: sample.file.type, size: bytes.byteLength, sha256: await sha256(bytes),
      path, synthetic: synthetic.has(sample.id),
    });
  }
  const manifest: Manifest = {
    kind: 'open-vision-classification-project', schemaVersion: 1,
    recipe: 'mobilenetv4-small-frozen-head-v1', labels: [...project.labels],
    epochs: project.epochs, learningRate: project.learningRate, images,
  };
  const data = strToU8(JSON.stringify(manifest));
  if (data.byteLength > MAX_MANIFEST_BYTES) fail('プロジェクトのメタデータが大きすぎます。');
  entries['manifest.json'] = data;
  const archive = zipSync(entries, { level: 0 });
  if (archive.byteLength > MAX_PROJECT_ARCHIVE_BYTES) fail('バックアップの上限512MiBを超えます。');
  return archive;
}

// Inspect the central directory *before* inflating anything. fflate's unzipSync
// alone can allocate attacker-declared uncompressed lengths (ZIP bombs).
export function inspectProjectZip(zip: Uint8Array): void {
  if (zip.byteLength > MAX_PROJECT_ARCHIVE_BYTES || zip.byteLength < 22) fail('ZIPの容量または構造が不正です。');
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  let eocd = -1;
  for (let p = zip.length - 22; p >= Math.max(0, zip.length - 65557); p--) {
    if (view.getUint32(p, true) === 0x06054b50 && p + 22 + view.getUint16(p + 20, true) === zip.length) {
      eocd = p; break;
    }
  }
  if (eocd < 0 || view.getUint16(eocd + 4, true) !== 0
      || view.getUint16(eocd + 6, true) !== 0
      || view.getUint16(eocd + 8, true) !== view.getUint16(eocd + 10, true)) fail('分割ZIPは使用できません。');
  const count = view.getUint16(eocd + 10, true);
  const size = view.getUint32(eocd + 12, true);
  const start = view.getUint32(eocd + 16, true);
  if (count < 1 || count > MAX_IMAGES + 1 || size === 0xffffffff || start === 0xffffffff
      || start + size > eocd) fail('ZIPの目録が不正です。');
  let offset = start;
  let total = 0;
  const names = new Set<string>();
  for (let i = 0; i < count; i++) {
    if (offset + 46 > start + size || view.getUint32(offset, true) !== 0x02014b50) fail('ZIPのファイル目録が破損しています。');
    const flags = view.getUint16(offset + 8, true);
    const method = view.getUint16(offset + 10, true);
    const compressed = view.getUint32(offset + 20, true);
    const uncompressed = view.getUint32(offset + 24, true);
    const n = view.getUint16(offset + 28, true);
    const ext = view.getUint16(offset + 30, true);
    const comment = view.getUint16(offset + 32, true);
    const end = offset + 46 + n + ext + comment;
    if (end > start + size || (flags & 1) || ![0, 8].includes(method)
        || compressed === 0xffffffff || uncompressed === 0xffffffff) fail('未対応のZIP形式です。');
    const name = new TextDecoder('utf-8', { fatal: true }).decode(zip.subarray(offset + 46, offset + 46 + n));
    const isManifest = name === 'manifest.json';
    if ((!isManifest && !/^images\/\d{3}\.bin$/.test(name))
        || names.has(name) || uncompressed > (isManifest ? MAX_MANIFEST_BYTES : MAX_FILE_BYTES)) {
      fail('ZIPのパス、重複、または展開後サイズが不正です。');
    }
    names.add(name);
    total += uncompressed;
    if (total > MAX_PROJECT_ARCHIVE_BYTES) fail('ZIPの展開後サイズが上限を超えています。');
    offset = end;
  }
  if (offset !== start + size || !names.has('manifest.json')) fail('ZIPの目録が不完全です。');
}

export async function importClassificationProject(zip: Uint8Array): Promise<ClassificationProject> {
  inspectProjectZip(zip);
  const files = unzipSync(zip);
  const manifestBytes = files['manifest.json'];
  if (!manifestBytes || manifestBytes.length > MAX_MANIFEST_BYTES) fail('manifest.json がありません。');
  let raw: unknown;
  try { raw = JSON.parse(strFromU8(manifestBytes)); }
  catch { fail('manifest.json がJSONとして不正です。'); }
  if (!object(raw) || raw.kind !== 'open-vision-classification-project'
      || raw.schemaVersion !== 1 || raw.recipe !== 'mobilenetv4-small-frozen-head-v1'
      || !validLabels(raw.labels) || !validSettings(raw.epochs, raw.learningRate)
      || !Array.isArray(raw.images) || raw.images.length > MAX_IMAGES
      || Object.keys(files).length !== raw.images.length + 1) fail('非対応または不正なプロジェクト形式です。');
  const labels = raw.labels;
  const samples: Sample[] = [];
  const seen = new Set<string>();
  const syntheticIds: string[] = [];
  for (let index = 0; index < raw.images.length; index++) {
    const item: unknown = raw.images[index];
    if (!object(item) || item.path !== pathFor(index) || !validId(item.id)
        || !validateImageName(item.name) || !Number.isInteger(item.label)
        || (item.label as number) < 0 || (item.label as number) >= labels.length
        || (item.split !== 'train' && item.split !== 'test')
        || typeof item.type !== 'string' || !IMAGE_TYPES.has(item.type)
        || !Number.isInteger(item.size) || (item.size as number) <= 0
        || (item.size as number) > MAX_FILE_BYTES
        || typeof item.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(item.sha256)
        || typeof item.synthetic !== 'boolean' || seen.has(item.id)) fail('画像目録が不正です。');
    seen.add(item.id);
    const bytes = files[item.path as string];
    if (!bytes || bytes.length !== item.size || await sha256(bytes) !== item.sha256) {
      fail('画像ファイルが欠損、変更、または破損しています。');
    }
    const type = item.type as string;
    const header = (s: string, at: number) => [...s].every((c, i) => bytes[at + i] === c.charCodeAt(0));
    if (!(type === 'image/png' && bytes[0] === 137 && header('PNG\r\n\x1a\n', 1)
        || type === 'image/jpeg' && bytes[0] === 255 && bytes[1] === 216
        || type === 'image/webp' && header('RIFF', 0) && header('WEBP', 8))) {
      fail('画像の形式と実データが一致しません。');
    }
    samples.push({ id: item.id, name: item.name as string, label: item.label as number,
      split: item.split as 'train' | 'test', file: new Blob([new Uint8Array(bytes)], { type }) });
    if (item.synthetic) syntheticIds.push(item.id);
  }
  return { labels, samples, epochs: raw.epochs as number,
    learningRate: raw.learningRate as number, syntheticIds };
}

export async function checkRestoredImageDimensions(project: ClassificationProject): Promise<void> {
  for (const sample of project.samples) {
    let bitmap: ImageBitmap;
    try { bitmap = await createImageBitmap(sample.file); }
    catch { fail('ZIP内の画像をデコードできません。'); }
    const width = bitmap.width, height = bitmap.height;
    bitmap.close();
    if (!width || !height || width * height > MAX_PIXELS) fail('ZIP内の画像が16メガピクセルを超えています。');
  }
}
