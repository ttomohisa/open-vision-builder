import type { RunSummary, Split } from '../protocol';

export const PROJECT_SCHEMA_VERSION = 1 as const;
export const PROJECT_TASK = 'classification' as const;
export const PROJECT_MODEL = 'mobilenetv4-conv-small' as const;
export const PROJECT_ID = 'classification-default' as const;

export const MAX_PROJECT_IMAGES = 64;
export const MAX_PROJECT_IMAGE_BYTES = 20 * 1024 * 1024;
export const MAX_PROJECT_MANIFEST_BYTES = 2 * 1024 * 1024;
export const MAX_PROJECT_MODEL_KIT_BYTES = 256 * 1024 * 1024;
export const MAX_PROJECT_BYTES =
  MAX_PROJECT_IMAGES * MAX_PROJECT_IMAGE_BYTES + MAX_PROJECT_MODEL_KIT_BYTES + MAX_PROJECT_MANIFEST_BYTES;

export const PROJECT_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

export interface ProjectSampleSnapshot {
  id: string;
  name: string;
  label: number;
  split: Split;
  synthetic: boolean;
  file: Blob;
}

export interface ProjectResultSnapshot {
  summary: RunSummary;
  archive: Uint8Array;
}

export interface ProjectSnapshot {
  labels: string[];
  epochs: string;
  learningRate: string;
  samples: ProjectSampleSnapshot[];
  result: ProjectResultSnapshot | null;
}

export interface ProjectBinaryRef {
  path: string;
  bytes: number;
  sha256: string;
  mime?: string;
}

export interface ProjectSampleManifest {
  id: string;
  name: string;
  label: number;
  split: Split;
  synthetic: boolean;
  file: ProjectBinaryRef;
}

export interface ProjectResultManifest {
  summary: RunSummary;
  archive: ProjectBinaryRef;
}

export interface ProjectManifest {
  schemaVersion: typeof PROJECT_SCHEMA_VERSION;
  task: typeof PROJECT_TASK;
  model: typeof PROJECT_MODEL;
  labels: string[];
  epochs: string;
  learningRate: string;
  samples: ProjectSampleManifest[];
  result: ProjectResultManifest | null;
}

export function isHexSha256(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}
