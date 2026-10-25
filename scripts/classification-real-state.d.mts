export function validatePlan(plan: unknown, manifest: unknown, manifestSha256: string): any[];
export function validatePreparedArtifact(prepared: unknown, modelSha256: string, builtSha256: string, plan: unknown, mode?: 'experiment' | 'app-startup'): void;
export function validateWorkerRun(request: unknown, summary: unknown, manifest: unknown, plan: unknown): void;
export function validateWorkerRequest(request: unknown, manifest: unknown, plan: unknown): void;
export function reserveOutput(path: string): Promise<void>;
export function extractExportedHead(bytes: Uint8Array, classes: number, featureSize?: number): { classes: number; featureSize: number; weights: number[]; bias: number[] };
