import { describe, expect, it } from 'vitest';
import { validateDetectorManifest } from './fixtures/detection-browser';

const shapes: Record<string, number[]> = {
  logits: [1, 300, 80], boxes: [1, 300, 4], bboxInput: [1, 300, 128],
  classInput: [1, 300, 128], residualCorners: [1, 300, 132], references: [1, 300, 4],
};
const limits = (name: string) => ({ atol: ['boxes', 'references'].includes(name) ? 2e-5 : 2e-4, rtol: 1e-4 });
const digest = '0123456789abcdef'.repeat(4);
const artifact = (file: string, bytes = 16) => ({ file, bytes, sha256: digest });

// Only metadata is needed: no checkpoint, tensor data, network or model execution.
function manifest(passed = false) {
  const parameters = [
    ...[0, 1, 2].flatMap(layer => ['weight', 'bias'].map(kind => ({
      parameter: `model.decoder.bbox_embed.2.layers.${layer}.${kind}`,
      initializers: [`base.model.decoder.bbox_embed.2.layers.${layer}.${kind}`],
      shape: kind === 'weight' ? [layer === 2 ? 132 : 128, 128] : [layer === 2 ? 132 : 128],
    }))),
    ...['weight', 'bias'].map(kind => ({
      parameter: `model.decoder.class_embed.2.${kind}`,
      initializers: [`base.model.decoder.class_embed.2.${kind}`],
      shape: kind === 'weight' ? [80, 128] : [80],
    })),
  ];
  return {
    schemaVersion: 1, id: 'dfine-nano', task: 'object-detection', passed,
    status: passed ? 'developer-investigation' : 'unqualified-intermediate-parity-failure',
    source: {
      modelId: 'ustc-community/dfine-nano-coco', revision: '066438d3d8f0da137a37b38fdf3368fd4afceced',
      sha256: '19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b', license: 'Apache-2.0',
    },
    input: { name: 'images', shape: [1, 3, 640, 640], dtype: 'float32' },
    outputs: Object.fromEntries(['logits', 'boxes'].map(name => [name, { shape: [...shapes[name]] }])),
    artifacts: {
      'detector.onnx': artifact('detector.onnx'), 'frozen-prefix.onnx': artifact('frozen-prefix.onnx'),
      headWeights: artifact('head-weights.json'),
    },
    exportRecipe: {
      schemaVersion: 1, id: 'dfine-nano-coco-final-head-v1',
      input: { name: 'images', shape: [1, 3, 640, 640] },
      outputs: ['logits', 'boxes'].map(name => ({ name, shape: [...shapes[name]] })), parameters,
    },
    labels: Array.from({ length: 80 }, (_, i) => `class ${i}`),
    licenses: ['LICENSE.txt', 'TRANSFORMERS_LICENSE.txt', 'NOTICE.txt', 'MODEL_CARD.md'],
    tolerances: Object.fromEntries(Object.keys(shapes).map(name => [name, limits(name)])),
    fixtures: [0, 37, 91].map(phase => ({
      id: `dfine-rgb-pattern-v1-phase-${phase}`, phase, inputSha256: digest,
      outputs: Object.fromEntries(Object.entries(shapes).map(([name, shape]) => [name, {
        ...artifact(`fixtures/pattern-${phase}/${name}.f32`, shape.reduce((a, b) => a * b, 4)),
        shape: [...shape], dtype: 'float32-le',
      }])),
      comparisons: Object.fromEntries(Object.entries(shapes).map(([name, shape]) => {
        const failed = !passed && phase === 37 && name === 'bboxInput';
        return [name, {
          passed: !failed, shape: [...shape], maxAbsoluteError: failed ? 0.001 : 0.000001,
          maxToleranceRatio: failed ? 2 : 0.5, ...limits(name),
        }];
      })),
    })),
  };
}

describe('detector investigation manifest boundary', () => {
  it('preserves a complete preparation failure outside the browser phase-0 fixture', () => {
    const source = manifest();
    expect(validateDetectorManifest(source)).toBe(source);
    expect(source.passed).toBe(false);
    expect(source.fixtures[1].comparisons.bboxInput.passed).toBe(false);
  });

  it('accepts a complete all-pass preparation without treating it as product readiness', () => {
    const source = manifest(true);
    expect(validateDetectorManifest(source)).toBe(source);
    expect(source.status).toBe('developer-investigation');
  });

  it.each(['passed', 'status', 'fixtures', 'source', 'artifacts', 'input', 'outputs', 'exportRecipe', 'tolerances'])(
    'rejects a manifest missing required %s metadata', key => {
      const source = manifest() as Record<string, unknown>;
      delete source[key];
      expect(() => validateDetectorManifest(source)).toThrow(/manifest/i);
    },
  );

  it('rejects an aggregate result or status that hides a measured feature failure', () => {
    const source = manifest();
    source.passed = true;
    source.status = 'developer-investigation';
    expect(() => validateDetectorManifest(source)).toThrow(/qualification/i);
    source.passed = false;
    expect(() => validateDetectorManifest(source)).toThrow(/qualification/i);
  });

  it('rejects missing or duplicated fixture phases', () => {
    const source = manifest();
    source.fixtures.pop();
    expect(() => validateDetectorManifest(source)).toThrow(/fixture/i);
    const duplicate = manifest();
    duplicate.fixtures[2] = structuredClone(duplicate.fixtures[1]);
    expect(() => validateDetectorManifest(duplicate)).toThrow(/fixture/i);
  });

  it('rejects a missing feature comparison or a nonboolean result', () => {
    const source = manifest();
    delete source.fixtures[1].comparisons.bboxInput;
    expect(() => validateDetectorManifest(source)).toThrow(/comparison/i);
    const malformed = manifest();
    Object.assign(malformed.fixtures[1].comparisons.bboxInput, { passed: 'false' });
    expect(() => validateDetectorManifest(malformed)).toThrow(/comparison/i);
  });

  it('rejects changed tolerances both globally and on an individual comparison', () => {
    const source = manifest();
    source.tolerances.residualCorners.atol = 0.01;
    expect(() => validateDetectorManifest(source)).toThrow(/tolerance/i);
    const changedRow = manifest();
    changedRow.fixtures[1].comparisons.bboxInput.rtol = 0.01;
    expect(() => validateDetectorManifest(changedRow)).toThrow(/tolerance/i);
  });

  it.each([NaN, Infinity, -1])('rejects invalid numeric evidence (%s)', value => {
    const source = manifest();
    source.fixtures[1].comparisons.bboxInput.maxToleranceRatio = value;
    expect(() => validateDetectorManifest(source)).toThrow(/comparison/i);
  });

  it('rejects a pass flag contradicting the measured maximum tolerance ratio', () => {
    const source = manifest();
    source.fixtures[1].comparisons.bboxInput.passed = true;
    expect(() => validateDetectorManifest(source)).toThrow(/comparison/i);
  });

  it.each(['logits', 'boxes', 'references'])('rejects a preparation failure in required %s', name => {
    const source = manifest();
    Object.assign(source.fixtures[2].comparisons[name], { passed: false, maxToleranceRatio: 2 });
    expect(() => validateDetectorManifest(source)).toThrow(/final-output|reference/i);
  });

  it('rejects incomplete or contradictory feature tensor metadata', () => {
    const source = manifest();
    delete source.fixtures[2].outputs.residualCorners;
    expect(() => validateDetectorManifest(source)).toThrow(/output/i);
    const changedShape = manifest();
    changedShape.fixtures[1].comparisons.bboxInput.shape = [1, 128, 300];
    expect(() => validateDetectorManifest(changedShape)).toThrow(/shape/i);
    const changedBytes = manifest();
    changedBytes.fixtures[1].outputs.bboxInput.bytes -= 4;
    expect(() => validateDetectorManifest(changedBytes)).toThrow(/bytes/i);
  });

  it('rejects an unknown model, invalid digest and artifact path escape', () => {
    const source = manifest();
    source.source.revision = 'another-revision';
    expect(() => validateDetectorManifest(source)).toThrow(/source/i);
    const changedDigest = manifest();
    changedDigest.artifacts['detector.onnx'].sha256 = 'missing';
    expect(() => validateDetectorManifest(changedDigest)).toThrow(/artifact/i);
    const escapedPath = manifest();
    escapedPath.artifacts['detector.onnx'].file = '../detector.onnx';
    expect(() => validateDetectorManifest(escapedPath)).toThrow(/artifact/i);
  });
});
