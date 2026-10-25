import { isDeepStrictEqual } from 'node:util';
import { dirname, resolve } from 'node:path';
import { lstat, mkdir } from 'node:fs/promises';
import { fromBinary } from '@bufbuild/protobuf';
import { ModelProtoSchema } from 'onnx-buf';

const classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'];
const hash = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
function require(condition, message) { if (!condition) throw new Error(message); }
function same(actual, expected, name) { require(isDeepStrictEqual(actual, expected), `Changed ${name}`); }

export function validatePreparedArtifact(prepared, modelSha256, builtSha256, plan, mode = 'experiment') {
  require(mode === 'experiment' || mode === 'app-startup', 'Unknown asset verification mode');
  require(hash(modelSha256) && prepared.artifact?.sha256 === modelSha256 && builtSha256 === modelSha256,
    'Prepared model manifest/build hash mismatch');
  require(prepared.source?.sha256 === plan.model.checkpointSha256 && prepared.source.revision === plan.model.sourceRevision,
    'Prepared checkpoint source differs from fixed plan');
  if (mode === 'experiment') require(modelSha256 === plan.model.onnxSha256, 'Prepared model differs from fixed measurement artifact');
}

export function validatePlan(plan, manifest, manifestSha256) {
  require(plan.schemaVersion === 1 && plan.id === 'classification-flowers-60-v1' && plan.releaseReady === false, 'Unsupported experiment');
  require(hash(manifestSha256) && plan.datasetManifestSha256 === manifestSha256, 'Dataset manifest hash mismatch');
  same(plan.classes, classes, 'class order'); same(manifest.classes, classes, 'dataset classes');
  same(plan.splits, { train: 30, tuning: 10, evaluation: 20 }, 'splits');
  same(plan.training, { featureSize: 1280, classes: 5, epochs: 40, learningRate: 0.01, batchSize: 30,
    initialization: 'zeros', optimizer: 'sgd', momentum: 0, weightDecay: 0, shuffle: false, augmentation: false, backend: 'wasm' }, 'training recipe');
  same(plan.preprocessing, { encodedInput: 'canonical-rgba-png', size: 224, resize: 'bilinear-half-pixel-stretch',
    mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], layout: 'NCHW', dtype: 'float32' }, 'preprocessing');
  same(plan.tolerances, { atol: 0.0002, rtol: 0.0001, inputMaxAbsoluteExclusive: 0.000001 }, 'numeric conditions');
  same(plan.criteria, { minimumTrainLossReduction: 0.2, minimumTrainAccuracy: 0.9, minimumEvaluationAccuracy: 0.7,
    maximumEvaluationAccuracyGap: 0.05, maximumEvaluationMacroF1Gap: 0.05 }, 'quality conditions');
  same(plan.budget, { maxCompletedRuns: 1, maxRuntimeSeconds: 600, maxActiveWorkMinutes: 90, automaticRetries: 0 }, 'run budget');
  require(plan.model?.id === 'timm/mobilenetv4_conv_small.e2400_r224_in1k' &&
    plan.model.sourceRevision === '331fb803779522b685cf942e15f914fb6741c1eb' &&
    plan.model.checkpointSha256 === '7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4' && hash(plan.model.onnxSha256), 'Unpinned model');
  require(manifest.schemaVersion === 1 && Array.isArray(manifest.images) && manifest.images.length === 60, 'Invalid dataset');
  const counts = classes.map(() => ({ train: 0, tuning: 0, evaluation: 0 }));
  const unique = Object.fromEntries(['id', 'creatorKey', 'flickrId', 'pngSha256', 'originalSha256', 'rgbaSha256'].map(key => [key, new Set()]));
  for (const item of manifest.images) {
    require(typeof item.id === 'string' && /^[a-z]+-[0-9]+$/.test(item.id), 'Unsafe image identifier');
    require(Number.isInteger(item.label) && item.label >= 0 && item.label < 5 && ['train', 'tuning', 'evaluation'].includes(item.split), 'Invalid label or split');
    require(item.id === `${classes[item.label]}-${item.flickrId}` && hash(item.pngSha256) && hash(item.originalSha256), 'Image identity mismatch');
    require(Number.isInteger(item.pngBytes) && item.pngBytes > 0 && item.pngBytes <= 20 * 1024 * 1024, 'Invalid image size');
    counts[item.label][item.split]++;
    for (const [key, seen] of Object.entries(unique)) {
      require(typeof item[key] === 'string' && item[key] && !seen.has(item[key]), `Repeated or missing ${key}`); seen.add(item[key]);
    }
  }
  counts.forEach(count => same(count, { train: 6, tuning: 2, evaluation: 4 }, 'per-class split'));
  return manifest.images.filter(item => item.split !== 'tuning');
}

export function validateWorkerRun(request, summary, manifest, plan) {
  validateWorkerRequest(request, manifest, plan);
  const tests = request.samples.filter(item => item.split === 'test');
  require(summary.epochs === 40 && summary.trainingCount === 30 && summary.testCount === 20 && summary.backend === 'wasm', 'Incomplete Worker run');
  require(summary.losses?.length === 40 && summary.losses.every(x => Number.isFinite(x) && x >= 0) &&
    Number.isFinite(summary.weightDeltaL2) && summary.weightDeltaL2 > 0, 'Invalid loss history or update');
  require(summary.predictions?.length === 20, 'Incomplete evaluation');
  let correct = 0;
  summary.predictions.forEach((prediction, index) => {
    require(prediction.id === tests[index].id && prediction.actual === tests[index].label && Number.isInteger(prediction.predicted) &&
      prediction.predicted >= 0 && prediction.predicted < 5 && prediction.scores?.length === 5 &&
      prediction.scores.every(x => Number.isFinite(x) && x >= 0 && x <= 1) && Math.abs(prediction.scores.reduce((a, b) => a + b, 0) - 1) < 1e-6,
    'Invalid actual Worker prediction');
    require(prediction.predicted === prediction.scores.indexOf(Math.max(...prediction.scores)), 'Worker predicted index mismatch');
    if (prediction.actual === prediction.predicted) correct++;
  });
  require(Number.isFinite(summary.accuracy) && Math.abs(summary.accuracy - correct / 20) < 1e-12, 'Worker accuracy mismatch');
}

export function validateWorkerRequest(request, manifest, plan) {
  const images = validatePlan(plan, manifest, plan.datasetManifestSha256);
  same(request.labels, plan.classes, 'actual Worker labels');
  require(request.epochs === 40 && request.learningRate === 0.01 && request.samples?.length === 50, 'Worker recipe/count mismatch');
  const ids = new Set();
  request.samples.forEach((sample, index) => {
    const item = images[index];
    require(typeof sample.id === 'string' && sample.id && !ids.has(sample.id), 'Duplicate Worker ID'); ids.add(sample.id);
    require(sample.imageId === item.id && sample.name === `${item.id}.png` && sample.fileSha256 === item.pngSha256 &&
      sample.label === item.label && sample.split === (item.split === 'train' ? 'train' : 'test'), 'Worker image/label/split/hash mismatch');
  });
}

export async function reserveOutput(path) {
  for (let cursor = resolve(path); ; cursor = dirname(cursor)) {
    const info = await lstat(cursor).catch(error => error.code === 'ENOENT' ? null : Promise.reject(error));
    require(!info?.isSymbolicLink(), 'Refusing a symlink experiment path');
    if (cursor === resolve(path)) require(!info, 'Experiment output exists; preserve the earlier result');
    if (dirname(cursor) === cursor) break;
  }
  await mkdir(dirname(resolve(path)), { recursive: true });
  await mkdir(resolve(path));
}

export function extractExportedHead(bytes, classes, featureSize = 1280) {
  const graph = fromBinary(ModelProtoSchema, bytes).graph;
  require(graph, 'Missing exported graph');
  function read(name, shape) {
    const matches = graph.initializer.filter(tensor => tensor.name === name);
    require(matches.length === 1, `Missing/duplicate ${name}`);
    const tensor = matches[0], count = shape.reduce((a, b) => a * b, 1);
    same(tensor.dims.map(Number), shape, 'head tensor shape');
    require(tensor.dataType === 1 && !tensor.externalData.length && tensor.dataLocation !== 1 && tensor.rawData.length === count * 4, 'Invalid embedded FP32 head');
    const view = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
    const values = Array.from({ length: count }, (_, index) => view.getFloat32(index * 4, true));
    require(values.every(Number.isFinite), 'Nonfinite exported head');
    return values;
  }
  return { classes, featureSize, weights: read('ovb.head.weight', [featureSize, classes]), bias: read('ovb.head.bias', [classes]) };
}
