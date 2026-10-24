export interface HeadWeights {
  weights: Float32Array;
  bias: Float32Array;
  featureSize: number;
  classes: number;
}

export interface ModelManifest {
  schemaVersion: 1;
  id: string;
  displayName: string;
  source: { modelId: string; revision: string; file: string; sha256: string; license: string; url: string };
  artifact: { file: string; sha256: string; bytes: number };
  input: { name: string; shape: number[]; mean: number[]; std: number[]; resize: string };
  features: { name: string; size: number };
  originalOutput: string;
  head: { nodeName: string; weightName: string; biasName: string };
  reference: { file: string; fixture: string };
  licenses: { licenseFile: string; noticeFile: string; modelCardFile: string };
}

export interface NativeReference {
  schemaVersion: 1;
  fixture: { name: string; width: number; height: number };
  native: { features: number[]; logits: number[] };
  tolerances: { atol: number; rtol: number };
  toolVersions: Record<string, string>;
}
