export type Split = 'train' | 'test';
export interface Sample { id: string; name: string; label: number; split: Split; file: Blob }
export interface RunRequest { type: 'run'; id: number; labels: string[]; samples: Sample[]; epochs: number; learningRate: number }
export type WorkerRequest = RunRequest | { type: 'cancel'; id: number };
export interface Prediction { id: string; actual: number; predicted: number; scores: number[] }
export interface RunSummary {
  epochs: number; losses: number[]; accuracy: number; predictions: Prediction[];
  trainingCount: number; testCount: number; durationMs: number;
  nativeMaxAbsError: number; exportMaxAbsError: number; weightDeltaL2: number;
  modelBytes: number; modelSha256: string; backend: string;
}
export type WorkerResponse =
  | { type: 'progress'; id: number; stage: string; fraction: number; loss?: number }
  | { type: 'result'; id: number; summary: RunSummary; archive: ArrayBuffer; model: ArrayBuffer; report: string }
  | { type: 'cancelled'; id: number }
  | { type: 'error'; id: number; message: string };
