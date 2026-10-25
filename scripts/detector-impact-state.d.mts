export function validateImpactPlan(value: unknown, manifestSha256: string): void;
export class ImpactProgress {
  constructor(trainIds: string[]);
  recordControl(step: number, passed: boolean): void;
  acceptStep(step: number, id: string): void;
  finish(): void;
  stop(reason: string): void;
  snapshot(): { status: string; completedSteps: number; controls: number[]; reason?: string };
}
