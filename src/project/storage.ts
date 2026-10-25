import {
  validateProjectManifest,
  validateProjectModelKit,
  validateProjectSnapshot,
  sha256Hex,
} from './backup';
import {
  PROJECT_ID,
  PROJECT_MODEL,
  PROJECT_SCHEMA_VERSION,
  PROJECT_TASK,
  type ProjectBinaryRef,
  type ProjectManifest,
  type ProjectResultManifest,
  type ProjectSampleManifest,
  type ProjectSnapshot,
} from './types';

const DB_NAME = 'open-vision-builder';
const DB_VERSION = 1;
const STORE_NAME = 'projects';
const PROJECTS_DIR = 'projects';
const OBJECTS_DIR = 'objects';

export interface ProjectStorageStatus {
  supported: boolean;
  usage: number | null;
  quota: number | null;
  persisted: boolean | null;
  reason?: string;
}

const blobHashes = new WeakMap<Blob, Promise<string>>();
let mutationQueue: Promise<unknown> = Promise.resolve();

function quotaMessage(error: unknown) {
  return error instanceof DOMException && error.name === 'QuotaExceededError'
    ? new Error('Project storage is full. Export a backup or free browser storage, then try again.')
    : error;
}

function storageSupport() {
  return typeof indexedDB !== 'undefined'
    && typeof navigator !== 'undefined'
    && !!navigator.storage
    && typeof navigator.storage.getDirectory === 'function';
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB transaction was aborted.'));
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed.'));
  });
}

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') return Promise.reject(new Error('IndexedDB is not available in this browser.'));
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Unable to open project storage.'));
    request.onblocked = () => reject(new Error('Project storage upgrade is blocked by another tab.'));
  });
}

async function readManifest(): Promise<ProjectManifest | null> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, 'readonly');
    const value = await requestResult(transaction.objectStore(STORE_NAME).get(PROJECT_ID));
    await transactionDone(transaction);
    return value === undefined ? null : validateProjectManifest(value);
  } finally {
    database.close();
  }
}

async function writeManifest(manifest: ProjectManifest) {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).put(manifest, PROJECT_ID);
    await transactionDone(transaction);
  } finally {
    database.close();
  }
}

async function deleteManifest() {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).delete(PROJECT_ID);
    await transactionDone(transaction);
  } finally {
    database.close();
  }
}

async function projectDirectory(create: boolean): Promise<FileSystemDirectoryHandle> {
  if (!storageSupport()) throw new Error('OPFS project storage is not available in this browser.');
  const root = await navigator.storage.getDirectory();
  const projects = await root.getDirectoryHandle(PROJECTS_DIR, { create });
  return projects.getDirectoryHandle(PROJECT_ID, { create });
}

async function objectsDirectory(create: boolean) {
  const project = await projectDirectory(create);
  return project.getDirectoryHandle(OBJECTS_DIR, { create });
}

async function hashBlob(blob: Blob) {
  let pending = blobHashes.get(blob);
  if (!pending) {
    pending = sha256Hex(blob);
    blobHashes.set(blob, pending);
  }
  return pending;
}

function binaryRef(path: string, bytes: number, sha256: string, mime?: string): ProjectBinaryRef {
  return { path, bytes, sha256, ...(mime ? { mime } : {}) };
}

async function buildManifest(snapshot: ProjectSnapshot) {
  validateProjectSnapshot(snapshot);
  const samples: ProjectSampleManifest[] = [];
  const sources = new Map<string, Blob | Uint8Array>();
  for (const sample of snapshot.samples) {
    const sha256 = await hashBlob(sample.file);
    sources.set(sha256, sample.file);
    samples.push({
      id: sample.id,
      name: sample.name,
      label: sample.label,
      split: sample.split,
      synthetic: sample.synthetic,
      file: binaryRef(`images/${sample.id}`, sample.file.size, sha256, sample.file.type),
    });
  }

  let result: ProjectResultManifest | null = null;
  if (snapshot.result) {
    validateProjectModelKit(snapshot.result.archive);
    const archive = snapshot.result.archive;
    const sha256 = await sha256Hex(archive);
    sources.set(sha256, archive);
    result = {
      summary: snapshot.result.summary,
      archive: binaryRef('result/model-kit.zip', archive.byteLength, sha256),
    };
  }

  const manifest: ProjectManifest = {
    schemaVersion: PROJECT_SCHEMA_VERSION,
    task: PROJECT_TASK,
    model: PROJECT_MODEL,
    labels: [...snapshot.labels],
    epochs: snapshot.epochs,
    learningRate: snapshot.learningRate,
    samples,
    result,
  };
  validateProjectManifest(manifest);
  return { manifest, sources };
}

async function writeObject(directory: FileSystemDirectoryHandle, sha256: string, source: Blob | Uint8Array) {
  try {
    const existing = await directory.getFileHandle(sha256);
    const file = await existing.getFile();
    const expected = source instanceof Blob ? source.size : source.byteLength;
    if (file.size === expected) return;
  } catch (error) {
    if (!(error instanceof DOMException) || error.name !== 'NotFoundError') throw error;
  }
  const handle = await directory.getFileHandle(sha256, { create: true });
  const writable = await handle.createWritable();
  try {
    if (source instanceof Blob) {
      await writable.write(source);
    } else {
      const owned = new Uint8Array(source.byteLength);
      owned.set(source);
      await writable.write(owned.buffer);
    }
    await writable.close();
  } catch (error) {
    await writable.abort().catch(() => undefined);
    throw error;
  }
}

async function garbageCollect(directory: FileSystemDirectoryHandle, retained: Set<string>) {
  for await (const [name, handle] of directory.entries()) {
    if (handle.kind === 'file' && !retained.has(name)) await directory.removeEntry(name);
  }
}

async function saveProjectNow(snapshot: ProjectSnapshot) {
  if (!storageSupport()) throw new Error('Local project storage is not available in this browser.');
  const { manifest, sources } = await buildManifest(snapshot);
  const directory = await objectsDirectory(true);
  for (const [sha256, source] of sources) await writeObject(directory, sha256, source);
  await writeManifest(manifest);
  await garbageCollect(directory, new Set(sources.keys()));
}

function enqueueMutation<T>(operation: () => Promise<T>): Promise<T> {
  const next = mutationQueue.then(operation, operation);
  mutationQueue = next.then(() => undefined, () => undefined);
  return next;
}

export function saveProject(snapshot: ProjectSnapshot): Promise<void> {
  return enqueueMutation(() => saveProjectNow(snapshot)).catch(error => {
    throw quotaMessage(error);
  });
}

async function readObject(directory: FileSystemDirectoryHandle, reference: ProjectBinaryRef) {
  let handle: FileSystemFileHandle;
  try {
    handle = await directory.getFileHandle(reference.sha256);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'NotFoundError') {
      throw new Error(`Saved project object is missing: ${reference.path}`);
    }
    throw error;
  }
  const file = await handle.getFile();
  if (file.size !== reference.bytes) throw new Error(`Saved project object has the wrong size: ${reference.path}`);
  if (await hashBlob(file) !== reference.sha256) throw new Error(`Saved project object checksum failed: ${reference.path}`);
  return file;
}

export async function loadProject(): Promise<ProjectSnapshot | null> {
  if (!storageSupport()) throw new Error('Local project storage is not available in this browser.');
  const manifest = await readManifest();
  if (!manifest) return null;
  let directory: FileSystemDirectoryHandle;
  try {
    directory = await objectsDirectory(false);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'NotFoundError') throw new Error('Saved project objects are missing.');
    throw error;
  }

  const samples = [];
  for (const sample of manifest.samples) {
    const file = await readObject(directory, sample.file);
    samples.push({
      id: sample.id,
      name: sample.name,
      label: sample.label,
      split: sample.split,
      synthetic: sample.synthetic,
      file: new Blob([file], { type: sample.file.mime }),
    });
  }

  let result: ProjectSnapshot['result'] = null;
  if (manifest.result) {
    const file = await readObject(directory, manifest.result.archive);
    const archive = new Uint8Array(await file.arrayBuffer());
    validateProjectModelKit(archive);
    result = { summary: manifest.result.summary, archive };
  }

  const snapshot: ProjectSnapshot = {
    labels: [...manifest.labels],
    epochs: manifest.epochs,
    learningRate: manifest.learningRate,
    samples,
    result,
  };
  validateProjectSnapshot(snapshot);
  return snapshot;
}

async function clearProjectNow() {
  if (typeof indexedDB !== 'undefined') await deleteManifest();
  if (typeof navigator === 'undefined' || !navigator.storage || typeof navigator.storage.getDirectory !== 'function') return;
  const root = await navigator.storage.getDirectory();
  try {
    const projects = await root.getDirectoryHandle(PROJECTS_DIR);
    await projects.removeEntry(PROJECT_ID, { recursive: true });
  } catch (error) {
    if (!(error instanceof DOMException) || error.name !== 'NotFoundError') throw error;
  }
}

export function clearProject(): Promise<void> {
  return enqueueMutation(clearProjectNow);
}

export async function getStorageStatus(): Promise<ProjectStorageStatus> {
  if (!storageSupport()) return {
    supported: false, usage: null, quota: null, persisted: null,
    reason: 'IndexedDB or OPFS is not available in this browser.',
  };
  let usage: number | null = null;
  let quota: number | null = null;
  let persisted: boolean | null = null;
  try {
    const estimate = await navigator.storage.estimate();
    usage = typeof estimate.usage === 'number' ? estimate.usage : null;
    quota = typeof estimate.quota === 'number' ? estimate.quota : null;
  } catch { /* Status is advisory; storage operations still decide availability. */ }
  try {
    if (typeof navigator.storage.persisted === 'function') persisted = await navigator.storage.persisted();
  } catch { /* Browsers may decline to expose persistence state. */ }
  return { supported: true, usage, quota, persisted };
}

export async function requestPersistentStorage(): Promise<boolean> {
  if (!storageSupport() || typeof navigator.storage.persist !== 'function') return false;
  return navigator.storage.persist();
}
