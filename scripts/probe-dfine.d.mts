export interface GridSampleProbeReport {
  operatorGate: 'PASS' | 'FAIL';
  detectorGates: 'NOT RUN';
  results: { name: string; passed: boolean }[];
  pytorch: { results: { name: string; passed: boolean }[] };
}
export function runGridSampleProbe(options?: { format?: 'esm' | 'cjs' }): Promise<GridSampleProbeReport>;
