import type { Sample } from './protocol';

// This is a *local working draft*, not the inference-model ZIP, portable ZIP
// backup, or a resumable training checkpoint. Single-key IndexedDB updates are
// atomic; we do not claim that browser storage survives site-data deletion.
const DB_NAME = 'open-vision-builder-classification-draft-v1';
const STORE = 'drafts';
const KEY = 'current';
const MAX_IMAGES = 64;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_TOTAL_BYTES = 512 * 1024 * 1024;
const TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const hashCache = new WeakMap<Blob, Promise<string>>();

export interface WorkingDraft {
  labels: string[];
  samples: Sample[];
  epochs: string; // Preserve an in-progress edit, including an invalid/empty value.
  learningRate: string;
  syntheticIds: string[];
}
interface SavedSample {
  id: string;
  name: string;
  label: number;
  split: 'train' | 'test';
  synthetic: boolean;
  type: string;
  size: number;
  sha256: string;
  file: Blob;
}
interface RecordV1 {
  schemaVersion: 1;
  revision: number;
  savedAt: number;
  labels: string[];
  epochs: string;
  learningRate: string;
  images: SavedSample[];
}
export interface LoadedDraft {
  draft: WorkingDraft;
  revision: number;
  savedAt: number;
}
export class DraftConflictError extends Error {
  constructor() {
    super('別のタブで保存内容が更新されました。このタブの自動保存を停止しました。バックアップを保存し、ページを再読み込みしてください。');
    this.name = 'DraftConflictError';
  }
}

function assertDraft(value: WorkingDraft) {
  if (!Array.isArray(value.labels) || value.labels.length < 2 || value.labels.length > 8
    || value.labels.some(x => typeof x !== 'string' || x.length > 80)
    || typeof value.epochs !== 'string' || value.epochs.length > 32
    || typeof value.learningRate !== 'string' || value.learningRate.length > 32
    || !Array.isArray(value.samples) || value.samples.length > MAX_IMAGES
    || !Array.isArray(value.syntheticIds)) {
    throw new Error('自動保存用のクラス、画像枚数、設定が不正です。');
  }
  let bytes = 0;
  const ids = new Set<string>();
  for (const sample of value.samples) {
    if (typeof sample.id !== 'string' || sample.id.length < 1 || sample.id.length > 128
      || /[\x00-\x1f\x7f]/.test(sample.id) || ids.has(sample.id)
      || typeof sample.name !== 'string' || sample.name.length < 1 || sample.name.length > 255
      || !Number.isInteger(sample.label) || sample.label < 0 || sample.label >= value.labels.length
      || (sample.split !== 'train' && sample.split !== 'test')
      || !(sample.file instanceof Blob) || !TYPES.has(sample.file.type)
      || sample.file.size < 1 || sample.file.size > MAX_IMAGE_BYTES) {
      throw new Error('自動保存用の画像情報が不正です。');
    }
    ids.add(sample.id);
    bytes += sample.file.size;
    if (bytes > MAX_TOTAL_BYTES) throw new Error('自動保存の上限512MiBを超えました。手動でZIPバックアップを保存してください。');
  }
  const synthetic = new Set<string>();
  for (const id of value.syntheticIds) {
    if (typeof id !== 'string' || !ids.has(id) || synthetic.has(id)) {
      throw new Error('自動保存用のサンプル画像IDが不正です。');
    }
    synthetic.add(id);
  }
}
async function hash(blob: Blob): Promise<string> {
  let ongoing = hashCache.get(blob);
  if (!ongoing) {
    ongoing = blob.arrayBuffer().then(bytes => crypto.subtle.digest('SHA-256', bytes))
      .then(digest => Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join(''));
    hashCache.set(blob, ongoing);
  }
  return ongoing;
}

function openDb(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') return Promise.reject(new Error('この環境ではIndexedDBを利用できません。ZIPバックアップを使ってください。'));
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => {
      const db = req.result;
      db.onversionchange = () => db.close();
      resolve(db);
    };
    req.onerror = () => reject(req.error ?? new Error('ブラウザ保存領域を開けませんでした。'));
    req.onblocked = () => reject(new Error('別のタブが保存領域を使用中です。'));
  });
}

async function readRaw(): Promise<unknown> {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(KEY);
      let value: unknown;
      req.onsuccess = () => { value = req.result; };
      tx.oncomplete = () => resolve(value);
      tx.onabort = () => reject(tx.error ?? new Error('保存内容の読み込みに失敗しました。'));
      tx.onerror = () => { /* onabort reports the transaction error */ };
    });
  } finally {
    db.close();
  }
}

function checkRecord(raw: unknown): asserts raw is RecordV1 {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('保存内容が壊れています。ZIPバックアップがあれば復元してください。');
  const value = raw as Partial<RecordV1>;
  if (value.schemaVersion !== 1 || !Number.isSafeInteger(value.revision) || (value.revision ?? -1) < 1
    || typeof value.savedAt !== 'number' || !Number.isFinite(value.savedAt)
    || !Array.isArray(value.labels) || value.labels.length < 2 || value.labels.length > 8
    || value.labels.some(x => typeof x !== 'string' || x.length > 80)
    || typeof value.epochs !== 'string' || value.epochs.length > 32
    || typeof value.learningRate !== 'string' || value.learningRate.length > 32
    || !Array.isArray(value.images) || value.images.length > MAX_IMAGES) {
    throw new Error('保存内容の形式に対応していません。ZIPバックアップがあれば復元してください。');
  }
}

export async function loadWorkingDraft(): Promise<LoadedDraft | null> {
  const raw = await readRaw();
  if (raw === undefined) return null;
  checkRecord(raw);
  const samples: Sample[] = [];
  const syntheticIds: string[] = [];
  for (const item of raw.images) {
    if (!item || typeof item.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(item.sha256)
      || typeof item.synthetic !== 'boolean' || !(item.file instanceof Blob)
      || item.file.size !== item.size || item.file.type !== item.type) {
      throw new Error('保存した画像の情報が壊れています。');
    }
    if (await hash(item.file) !== item.sha256) throw new Error('保存した画像のSHA-256が一致しません。');
    samples.push({ id: item.id, name: item.name, label: item.label, split: item.split, file: item.file });
    if (item.synthetic) syntheticIds.push(item.id);
  }
  const draft: WorkingDraft = {
    labels: [...raw.labels], epochs: raw.epochs, learningRate: raw.learningRate,
    samples, syntheticIds,
  };
  assertDraft(draft);
  for (const sample of draft.samples) {
    let bitmap: ImageBitmap;
    try { bitmap = await createImageBitmap(sample.file); }
    catch { throw new Error('自動保存した画像を開けませんでした。ZIPバックアップがあれば復元してください。'); }
    const pixels = bitmap.width * bitmap.height;
    bitmap.close();
    if (!pixels || pixels > 16_000_000) throw new Error('保存画像の解像度が上限を超えています。');
  }
  return { draft, revision: raw.revision, savedAt: raw.savedAt };
}

async function serializeDraft(draft: WorkingDraft): Promise<Omit<RecordV1, 'revision' | 'savedAt'>> {
  assertDraft(draft);
  const synthetic = new Set(draft.syntheticIds);
  // Hash one image at a time. Reading 64 potentially large Blobs concurrently
  // would create an unnecessary temporary memory spike on small PCs.
  const images: SavedSample[] = [];
  for (const sample of draft.samples) {
    images.push({
      id: sample.id, name: sample.name, label: sample.label,
      split: sample.split, synthetic: synthetic.has(sample.id),
      type: sample.file.type, size: sample.file.size, file: sample.file,
      sha256: await hash(sample.file),
    });
  }
  return {
    schemaVersion: 1, labels: [...draft.labels],
    epochs: draft.epochs, learningRate: draft.learningRate, images,
  };
}

/**
 * Compare-and-swap on the previous revision inside the *same* IndexedDB
 * readwrite transaction. Competing tabs cannot silently overwrite each other.
 * On abort (including quota exhaustion), the previous record stays intact.
 */
export async function saveWorkingDraft(draft: WorkingDraft, expectedRevision: number | null): Promise<LoadedDraft> {
  const serialized = await serializeDraft(draft); // no awaits inside the IDB transaction
  const db = await openDb();
  try {
    const result = await new Promise<{ revision: number; savedAt: number }>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite', { durability: 'strict' });
      const req = tx.objectStore(STORE).get(KEY);
      let conflict = false;
      let writeError: DOMException | null = null;
      let result: { revision: number; savedAt: number } | null = null;
      req.onsuccess = () => {
        const previous: unknown = req.result;
        const revision = previous === undefined ? null
          : typeof previous === 'object' && previous !== null && 'revision' in previous
            ? (previous as { revision: unknown }).revision : -1;
        if (revision !== expectedRevision) {
          conflict = true;
          tx.abort();
          return;
        }
        result = { revision: (expectedRevision ?? 0) + 1, savedAt: Date.now() };
        tx.objectStore(STORE).put({ ...serialized, ...result }, KEY);
      };
      tx.oncomplete = () => {
        if (!result) reject(new Error('保存が完了しませんでした。'));
        else resolve(result);
      };
      tx.onabort = () => {
        if (conflict) reject(new DraftConflictError());
        else reject(writeError ?? tx.error ?? new Error('保存処理が中断されました。'));
      };
      tx.onerror = (event) => {
        writeError = (event.target as IDBRequest).error;
      };
    });
    return { draft, ...result };
  } finally {
    db.close();
  }
}

export async function describeStorage(): Promise<{ used: number | null; quota: number | null; persisted: boolean | null }> {
  if (!navigator.storage) return { used: null, quota: null, persisted: null };
  const estimate = await navigator.storage.estimate().catch(() => null);
  const persisted = typeof navigator.storage.persisted === 'function'
    ? await navigator.storage.persisted().catch(() => null) : null;
  return { used: estimate?.usage ?? null, quota: estimate?.quota ?? null, persisted };
}

export async function requestPersistentStorage(): Promise<boolean> {
  if (!navigator.storage || typeof navigator.storage.persist !== 'function') return false;
  return navigator.storage.persist();
}
