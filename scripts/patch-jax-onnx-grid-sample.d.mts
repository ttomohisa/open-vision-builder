export interface GridSamplePatchReport {
  package: string;
  version: string;
  patchId: string;
  status: 'applied' | 'already-applied' | 'verified';
  licenseSha256: string;
  files: { path: string; upstreamSha256: string; patchedSha256: string }[];
}
export function applyGridSamplePatch(options?: { packageDirectory?: string; check?: boolean }): GridSamplePatchReport;
