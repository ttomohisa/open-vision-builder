import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import type { Prediction, RunSummary, Split } from '../protocol';
import {
  MAX_PROJECT_BYTES,
  MAX_PROJECT_IMAGE_BYTES,
  MAX_PROJECT_IMAGES,
  MAX_PROJECT_MANIFEST_BYTES,
  MAX_PROJECT_MODEL_KIT_BYTES,
  PROJECT_IMAGE_TYPES,
  PROJECT_MODEL,
  PROJECT_SCHEMA_VERSION,
  PROJECT_TASK,
  isHexSha256,
  type ProjectBinaryRef,
  type ProjectManifest,
  type ProjectResultSnapshot,
  type ProjectSampleManifest,
  type ProjectSnapshot,
} from './types';

const MAX_BACKUP_ENTRIES = MAX_PROJECT_IMAGES + 3;
const MAX_MODEL_KIT_ENTRIES = 64;
const MAX_MODEL_KIT_EXPANDED_BYTES = 512 * 1024 * 1024;
const MODEL_KIT_REQUIRED = ['model.onnx', 'labels.json', 'manifest.json', 'report.json'] as const;
const textDecoder = new TextDecoder();

type ZipEntry = {
  name: string;
  method: number;
  compressedBytes: number;
  uncompressedBytes: number;
  localOffset: number;
  data: Uint8Array;
};

function fail(message: string): never {
  throw new Error(message);
}

function isSafeEntryName(name: string) {
  if (!name || name.startsWith('/') || name.includes('\\') || name.includes('\0')) return false;
  const segments = name.split('/');
  return !segments.some(segment => segment === '' || segment === '.' || segment === '..');
}

function findEocd(bytes: Uint8Array) {
  if (bytes.byteLength < 22) fail('Invalid ZIP: end record is missing.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const minimum = Math.max(0, bytes.byteLength - 65_557);
  for (let offset = bytes.byteLength - 22; offset >= minimum; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) return offset;
  }
  return fail('Invalid ZIP: end record is missing.');
}

function inspectZip(bytes: Uint8Array, options: {
  allowedMethods: Set<number>;
  maxEntries: number;
  maxExpandedBytes: number;
  requireStored?: boolean;
}) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = findEocd(bytes);
  const disk = view.getUint16(eocd + 4, true);
  const centralDisk = view.getUint16(eocd + 6, true);
  const entriesOnDisk = view.getUint16(eocd + 8, true);
  const entries = view.getUint16(eocd + 10, true);
  const centralBytes = view.getUint32(eocd + 12, true);
  const centralOffset = view.getUint32(eocd + 16, true);
  const commentBytes = view.getUint16(eocd + 20, true);
  if (disk !== 0 || centralDisk !== 0 || entriesOnDisk !== entries) fail('Multi-disk ZIP archives are not supported.');
  if (entries === 0 || entries > options.maxEntries) fail('ZIP entry count is outside the supported limit.');
  if (entries === 0xffff || centralBytes === 0xffffffff || centralOffset === 0xffffffff) fail('ZIP64 archives are not supported.');
  if (eocd + 22 + commentBytes !== bytes.byteLength) fail('Invalid ZIP trailing data.');
  if (centralOffset + centralBytes > eocd) fail('Invalid ZIP central directory bounds.');

  const seen = new Set<string>();
  const result: ZipEntry[] = [];
  let totalExpanded = 0;
  let offset = centralOffset;
  for (let index = 0; index < entries; index += 1) {
    if (offset + 46 > eocd || view.getUint32(offset, true) !== 0x02014b50) fail('Invalid ZIP central directory entry.');
    const flags = view.getUint16(offset + 8, true);
    const method = view.getUint16(offset + 10, true);
    const compressedBytes = view.getUint32(offset + 20, true);
    const uncompressedBytes = view.getUint32(offset + 24, true);
    const nameBytes = view.getUint16(offset + 28, true);
    const extraBytes = view.getUint16(offset + 30, true);
    const entryCommentBytes = view.getUint16(offset + 32, true);
    const startDisk = view.getUint16(offset + 34, true);
    const localOffset = view.getUint32(offset + 42, true);
    const end = offset + 46 + nameBytes + extraBytes + entryCommentBytes;
    if (end > eocd) fail('Invalid ZIP central directory entry bounds.');
    if ((flags & 1) !== 0) fail('Encrypted ZIP entries are not supported.');
    if (startDisk !== 0) fail('Multi-disk ZIP entries are not supported.');
    if (!options.allowedMethods.has(method)) fail(options.requireStored
      ? 'Project backup ZIP entries must use stored (uncompressed) encoding.'
      : 'Unsupported model-kit ZIP compression method.');
    if (options.requireStored && (method !== 0 || compressedBytes !== uncompressedBytes)) {
      fail('Project backup ZIP entries must use stored (uncompressed) encoding.');
    }
    const name = textDecoder.decode(bytes.subarray(offset + 46, offset + 46 + nameBytes));
    if (!isSafeEntryName(name)) fail('Unsafe ZIP entry path.');
    if (seen.has(name)) fail('Duplicate ZIP entry.');
    seen.add(name);
    totalExpanded += uncompressedBytes;
    if (totalExpanded > options.maxExpandedBytes) fail('ZIP expanded size exceeds the supported limit.');

    if (localOffset + 30 > centralOffset || view.getUint32(localOffset, true) !== 0x04034b50) fail('Invalid ZIP local entry.');
    const localFlags = view.getUint16(localOffset + 6, true);
    const localMethod = view.getUint16(localOffset + 8, true);
    const localNameBytes = view.getUint16(localOffset + 26, true);
    const localExtraBytes = view.getUint16(localOffset + 28, true);
    if ((localFlags & 1) !== 0 || localMethod !== method) fail('ZIP central/local metadata mismatch.');
    const localName = textDecoder.decode(bytes.subarray(localOffset + 30, localOffset + 30 + localNameBytes));
    if (localName !== name) fail('ZIP central/local filename mismatch.');
    const dataStart = localOffset + 30 + localNameBytes + localExtraBytes;
    if (dataStart + compressedBytes > centralOffset) fail('Invalid ZIP entry data bounds.');
    result.push({
      name, method, compressedBytes, uncompressedBytes, localOffset,
      data: bytes.subarray(dataStart, dataStart + compressedBytes),
    });
    offset = end;
  }
  if (offset !== centralOffset + centralBytes) fail('ZIP central directory size mismatch.');
  return result;
}

export async function sha256Hex(data: Blob | Uint8Array): Promise<string> {
  const input = data instanceof Blob ? new Uint8Array(await data.arrayBuffer()) : data;
  const copy = new Uint8Array(input.byteLength);
  copy.set(input);
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validateLabels(value: unknown): string[] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 8) fail('Project labels must contain 2 to 8 classes.');
  return value.map(label => {
    if (typeof label !== 'string' || label.length > 80) fail('Invalid project class label.');
    return label;
  });
}

function validateSetting(value: unknown, name: string) {
  if (typeof value !== 'string' || value.length > 32) fail(`Invalid project ${name} setting.`);
  if (value && !Number.isFinite(Number(value))) fail(`Invalid project ${name} setting.`);
  return value;
}

function finiteNumber(value: unknown, name: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`Invalid result ${name}.`);
  return value;
}

function integer(value: unknown, name: string, min = 0, max = Number.MAX_SAFE_INTEGER) {
  if (!Number.isInteger(value) || (value as number) < min || (value as number) > max) fail(`Invalid result ${name}.`);
  return value as number;
}

function validatePrediction(value: unknown, labels: number): Prediction {
  if (!isRecord(value) || typeof value.id !== 'string' || value.id.length > 128) fail('Invalid result prediction.');
  const actual = integer(value.actual, 'prediction actual', 0, labels - 1);
  const predicted = integer(value.predicted, 'prediction predicted', 0, labels - 1);
  if (!Array.isArray(value.scores) || value.scores.length !== labels) fail('Invalid result prediction scores.');
  const scores = value.scores.map((score, index) => finiteNumber(score, `prediction score ${index}`));
  return { id: value.id, actual, predicted, scores };
}

function validateSummary(value: unknown, labelCount: number): RunSummary {
  if (!isRecord(value)) fail('Invalid result summary.');
  if (!Array.isArray(value.losses) || value.losses.length > 200) fail('Invalid result losses.');
  const losses = value.losses.map((loss, index) => finiteNumber(loss, `loss ${index}`));
  if (!Array.isArray(value.predictions) || value.predictions.length > MAX_PROJECT_IMAGES) fail('Invalid result predictions.');
  const predictions = value.predictions.map(item => validatePrediction(item, labelCount));
  const modelSha256 = value.modelSha256;
  if (!isHexSha256(modelSha256)) fail('Invalid result model SHA-256.');
  if (typeof value.backend !== 'string' || !value.backend || value.backend.length > 80) fail('Invalid result backend.');
  return {
    epochs: integer(value.epochs, 'epochs', 1, 200),
    losses,
    accuracy: finiteNumber(value.accuracy, 'accuracy'),
    predictions,
    trainingCount: integer(value.trainingCount, 'training count', 0, MAX_PROJECT_IMAGES),
    testCount: integer(value.testCount, 'test count', 0, MAX_PROJECT_IMAGES),
    durationMs: finiteNumber(value.durationMs, 'duration'),
    nativeMaxAbsError: finiteNumber(value.nativeMaxAbsError, 'native error'),
    exportMaxAbsError: finiteNumber(value.exportMaxAbsError, 'export error'),
    weightDeltaL2: finiteNumber(value.weightDeltaL2, 'weight delta'),
    modelBytes: integer(value.modelBytes, 'model bytes', 1, MAX_PROJECT_MODEL_KIT_BYTES),
    modelSha256,
    backend: value.backend,
  };
}

function validateBinaryRef(value: unknown, expectedPath: string, maxBytes: number, mime?: string): ProjectBinaryRef {
  if (!isRecord(value) || value.path !== expectedPath) fail('Invalid project binary path.');
  const bytes = integer(value.bytes, 'binary bytes', 1, maxBytes);
  if (!isHexSha256(value.sha256)) fail('Invalid project binary SHA-256.');
  if (mime !== undefined && value.mime !== mime) fail('Invalid project binary MIME type.');
  if (value.mime !== undefined && typeof value.mime !== 'string') fail('Invalid project binary MIME type.');
  return { path: expectedPath, bytes, sha256: value.sha256, ...(typeof value.mime === 'string' ? { mime: value.mime } : {}) };
}

function validateSampleMetadata(value: unknown, labels: number): ProjectSampleManifest {
  if (!isRecord(value)) fail('Invalid project sample.');
  if (typeof value.id !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/.test(value.id)) fail('Invalid project sample id.');
  if (typeof value.name !== 'string' || !value.name || value.name.length > 255) fail('Invalid project sample name.');
  const label = integer(value.label, 'sample label', 0, labels - 1);
  if (value.split !== 'train' && value.split !== 'test') fail('Invalid project sample split.');
  if (typeof value.synthetic !== 'boolean') fail('Invalid project sample synthetic flag.');
  if (!isRecord(value.file) || typeof value.file.mime !== 'string' || !PROJECT_IMAGE_TYPES.has(value.file.mime)) fail('Invalid project image MIME type.');
  return {
    id: value.id,
    name: value.name,
    label,
    split: value.split as Split,
    synthetic: value.synthetic,
    file: validateBinaryRef(value.file, `images/${value.id}`, MAX_PROJECT_IMAGE_BYTES, value.file.mime),
  };
}

export function validateProjectManifest(value: unknown): ProjectManifest {
  if (!isRecord(value)) fail('Invalid project manifest.');
  if (value.schemaVersion !== PROJECT_SCHEMA_VERSION) fail('Unsupported project schema version.');
  if (value.task !== PROJECT_TASK) fail('Unsupported project task.');
  if (value.model !== PROJECT_MODEL) fail('Unsupported project model.');
  const labels = validateLabels(value.labels);
  const epochs = validateSetting(value.epochs, 'epochs');
  const learningRate = validateSetting(value.learningRate, 'learning rate');
  if (!Array.isArray(value.samples) || value.samples.length > MAX_PROJECT_IMAGES) fail('Project images exceed the 64 image limit.');
  const samples = value.samples.map(item => validateSampleMetadata(item, labels.length));
  if (new Set(samples.map(item => item.id)).size !== samples.length) fail('Duplicate project sample id.');
  let result: ProjectManifest['result'] = null;
  if (value.result !== null && value.result !== undefined) {
    if (!isRecord(value.result)) fail('Invalid project result.');
    result = {
      summary: validateSummary(value.result.summary, labels.length),
      archive: validateBinaryRef(value.result.archive, 'result/model-kit.zip', MAX_PROJECT_MODEL_KIT_BYTES),
    };
  }
  return { schemaVersion: 1, task: PROJECT_TASK, model: PROJECT_MODEL, labels, epochs, learningRate, samples, result };
}

function validateSnapshotShape(snapshot: ProjectSnapshot) {
  const labels = validateLabels(snapshot.labels);
  validateSetting(snapshot.epochs, 'epochs');
  validateSetting(snapshot.learningRate, 'learning rate');
  if (!Array.isArray(snapshot.samples) || snapshot.samples.length > MAX_PROJECT_IMAGES) fail('Project images exceed the 64 image limit.');
  const ids = new Set<string>();
  for (const sample of snapshot.samples) {
    if (typeof sample.id !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/.test(sample.id) || ids.has(sample.id)) fail('Invalid or duplicate project sample id.');
    ids.add(sample.id);
    if (typeof sample.name !== 'string' || !sample.name || sample.name.length > 255) fail('Invalid project sample name.');
    if (!Number.isInteger(sample.label) || sample.label < 0 || sample.label >= labels.length) fail('Invalid project sample label.');
    if (sample.split !== 'train' && sample.split !== 'test') fail('Invalid project sample split.');
    if (typeof sample.synthetic !== 'boolean') fail('Invalid project sample synthetic flag.');
    if (!(sample.file instanceof Blob) || !PROJECT_IMAGE_TYPES.has(sample.file.type) || sample.file.size < 1 || sample.file.size > MAX_PROJECT_IMAGE_BYTES) fail('Invalid project image file.');
  }
  if (snapshot.result) {
    validateSummary(snapshot.result.summary, labels.length);
    if (!(snapshot.result.archive instanceof Uint8Array) || snapshot.result.archive.byteLength < 1 || snapshot.result.archive.byteLength > MAX_PROJECT_MODEL_KIT_BYTES) {
      fail('Invalid project model kit.');
    }
  }
}

function validateModelKit(archive: Uint8Array) {
  if (archive.byteLength > MAX_PROJECT_MODEL_KIT_BYTES) fail('Project model kit exceeds the supported size.');
  inspectZip(archive, { allowedMethods: new Set([0, 8]), maxEntries: MAX_MODEL_KIT_ENTRIES, maxExpandedBytes: MAX_MODEL_KIT_EXPANDED_BYTES });
  let files: Record<string, Uint8Array>;
  try { files = unzipSync(archive); }
  catch { return fail('Invalid project model-kit ZIP.'); }
  for (const name of MODEL_KIT_REQUIRED) {
    if (!files[name]?.byteLength) fail(`Project model kit is missing ${name}.`);
  }
}

async function checkedRef(path: string, data: Blob | Uint8Array, maxBytes: number, mime?: string): Promise<ProjectBinaryRef> {
  const bytes = data instanceof Blob ? data.size : data.byteLength;
  if (bytes < 1 || bytes > maxBytes) fail('Project binary size exceeds the supported limit.');
  return { path, bytes, sha256: await sha256Hex(data), ...(mime ? { mime } : {}) };
}

export async function createProjectBackup(snapshot: ProjectSnapshot): Promise<Uint8Array> {
  validateSnapshotShape(snapshot);
  const files: Record<string, Uint8Array> = {};
  const samples: ProjectSampleManifest[] = [];
  let total = 0;
  for (const sample of snapshot.samples) {
    const data = new Uint8Array(await sample.file.arrayBuffer());
    total += data.byteLength;
    if (total > MAX_PROJECT_BYTES) fail('Project backup exceeds the supported total size.');
    const path = `images/${sample.id}`;
    files[path] = data;
    samples.push({
      id: sample.id, name: sample.name, label: sample.label, split: sample.split, synthetic: sample.synthetic,
      file: await checkedRef(path, data, MAX_PROJECT_IMAGE_BYTES, sample.file.type),
    });
  }

  let result: ProjectManifest['result'] = null;
  if (snapshot.result) {
    validateModelKit(snapshot.result.archive);
    total += snapshot.result.archive.byteLength;
    if (total > MAX_PROJECT_BYTES) fail('Project backup exceeds the supported total size.');
    const archive = new Uint8Array(snapshot.result.archive);
    files['result/model-kit.zip'] = archive;
    result = {
      summary: validateSummary(snapshot.result.summary, snapshot.labels.length),
      archive: await checkedRef('result/model-kit.zip', archive, MAX_PROJECT_MODEL_KIT_BYTES),
    };
  }

  const manifest: ProjectManifest = {
    schemaVersion: 1, task: PROJECT_TASK, model: PROJECT_MODEL,
    labels: [...snapshot.labels], epochs: snapshot.epochs, learningRate: snapshot.learningRate, samples, result,
  };
  const manifestBytes = strToU8(JSON.stringify(manifest, null, 2));
  if (manifestBytes.byteLength > MAX_PROJECT_MANIFEST_BYTES) fail('Project manifest exceeds the supported size.');
  files['project.json'] = manifestBytes;
  return zipSync(files, { level: 0 });
}

export async function parseProjectBackup(bytes: Uint8Array): Promise<ProjectSnapshot> {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 22 || bytes.byteLength > MAX_PROJECT_BYTES + MAX_PROJECT_MANIFEST_BYTES + 8 * 1024 * 1024) {
    fail('Project backup ZIP size is outside the supported limit.');
  }
  const entries = inspectZip(bytes, {
    allowedMethods: new Set([0]), maxEntries: MAX_BACKUP_ENTRIES,
    maxExpandedBytes: MAX_PROJECT_BYTES + MAX_PROJECT_MANIFEST_BYTES, requireStored: true,
  });
  const files = new Map(entries.map(entry => [entry.name, entry.data]));
  const manifestBytes = files.get('project.json');
  if (!manifestBytes || manifestBytes.byteLength > MAX_PROJECT_MANIFEST_BYTES) fail('Project backup is missing a valid project.json.');
  let raw: unknown;
  try { raw = JSON.parse(strFromU8(manifestBytes)); }
  catch { return fail('Project manifest is not valid JSON.'); }
  const manifest = validateProjectManifest(raw);

  const expected = new Set(['project.json', ...manifest.samples.map(item => item.file.path)]);
  if (manifest.result) expected.add(manifest.result.archive.path);
  if (expected.size !== files.size || entries.some(entry => !expected.has(entry.name))) fail('Project backup contains unexpected ZIP entries.');

  const samples = [];
  for (const sample of manifest.samples) {
    const data = files.get(sample.file.path);
    if (!data || data.byteLength !== sample.file.bytes) fail('Project image size does not match its manifest.');
    if (await sha256Hex(data) !== sample.file.sha256) fail('Project image checksum does not match its manifest.');
    samples.push({
      id: sample.id, name: sample.name, label: sample.label, split: sample.split, synthetic: sample.synthetic,
      file: new Blob([new Uint8Array(data)], { type: sample.file.mime! }),
    });
  }

  let result: ProjectResultSnapshot | null = null;
  if (manifest.result) {
    const archive = files.get(manifest.result.archive.path);
    if (!archive || archive.byteLength !== manifest.result.archive.bytes) fail('Project model kit size does not match its manifest.');
    if (await sha256Hex(archive) !== manifest.result.archive.sha256) fail('Project model kit checksum does not match its manifest.');
    const copy = new Uint8Array(archive);
    validateModelKit(copy);
    result = { summary: manifest.result.summary, archive: copy };
  }

  return {
    labels: [...manifest.labels], epochs: manifest.epochs, learningRate: manifest.learningRate,
    samples, result,
  };
}
