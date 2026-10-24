/**
 * Explicit correction for @jax-js/onnx@0.1.2, not an install lifecycle hook.
 * Only the four GridSample neighbor-index lines are changed. Both distributed
 * module formats and the original license must match before any file is written.
 * Upgrade/removal requires requalification against the independent reference.
 *
 * Source: https://unpkg.com/@jax-js/onnx@0.1.2/dist/index.js
 * Upstream fragments below retain their original MIT license:
 *
 * MIT License
 *
 * Copyright (c) 2025 Eric Zhang
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageName = '@jax-js/onnx';
const version = '0.1.2';
const licenseSha256 = 'b9de91fdc4f72bb4e49ad8b57306163bea46f6da7293c08f3b28a922727a952c';
const targets = [
  {
    path: 'dist/index.js', namespace: 'numpy',
    upstreamSha256: '7ef273af056e1021d668b873c5200027f6a3b326cef50972189e7446a393a7a1',
    patchedSha256: '362da04f6925eebebf6c2c02d938e3d4f23a610e2c262427b98c50da59b24f51',
  },
  {
    path: 'dist/index.cjs', namespace: '_jax_js_jax.numpy',
    upstreamSha256: '166276ea065e41b88ed6cee935451903f4e0c4db42f8c0c7709aa8ef60d07213',
    patchedSha256: '22cd8f1b00ad125d40dbf4cde7af35c169eccf48b3abcc3529966047ddafa893',
  },
];
const digest = content => createHash('sha256').update(content).digest('hex');

function correctedSource(source, namespace) {
  // Preserve a reference to each raw floor for its neighbor. Clipping x0/y0
  // first changes a raw -1 neighbor from pixel 0 to pixel 1 at top/left edges.
  const before = [];
  const after = [];
  for (const [axis, size] of [['x', 'Win'], ['y', 'Hin']]) {
    before.push(`\tconst ${axis}0c = ${namespace}.astype(${namespace}.clip(${axis}0f, 0, ${size} - 1), ${namespace}.int32);`);
    after.push(`\tconst ${axis}0c = ${namespace}.astype(${namespace}.clip(${axis}0f.ref, 0, ${size} - 1), ${namespace}.int32);`);
  }
  for (const [axis, size] of [['x', 'Win'], ['y', 'Hin']]) {
    before.push(`\tconst ${axis}1c = ${namespace}.clip(${namespace}.add(${axis}0c.ref, 1), 0, ${size} - 1);`);
    after.push(`\tconst ${axis}1c = ${namespace}.astype(${namespace}.clip(${namespace}.add(${axis}0f, 1), 0, ${size} - 1), ${namespace}.int32);`);
  }
  const original = before.join('\n');
  if (source.indexOf(original) < 0 || source.indexOf(original) !== source.lastIndexOf(original)) {
    throw new Error('Expected exactly one upstream GridSample index block.');
  }
  return source.replace(original, after.join('\n'));
}

export function applyGridSamplePatch({
  packageDirectory = fileURLToPath(new URL('../node_modules/@jax-js/onnx/', import.meta.url)),
  check = false,
} = {}) {
  const manifest = JSON.parse(readFileSync(join(packageDirectory, 'package.json'), 'utf8'));
  if (manifest.name !== packageName || manifest.version !== version) {
    throw new Error(`Expected ${packageName}@${version}; found ${manifest.name}@${manifest.version}. Re-audit the GridSample patch.`);
  }
  if (digest(readFileSync(join(packageDirectory, 'LICENSE'))) !== licenseSha256) {
    throw new Error('Unrecognized upstream LICENSE; refusing to modify the package.');
  }

  // Complete all source and output validation before either bundle is changed.
  const plans = targets.map(target => {
    const filename = join(packageDirectory, target.path);
    const source = readFileSync(filename, 'utf8');
    const hash = digest(source);
    if (hash !== target.upstreamSha256 && hash !== target.patchedSha256) {
      throw new Error(`Unrecognized ${target.path} SHA-256 ${hash}; refusing to modify the package.`);
    }
    const prepared = hash === target.patchedSha256;
    const output = prepared ? source : correctedSource(source, target.namespace);
    if (digest(output) !== target.patchedSha256) throw new Error(`Incorrect GridSample patch result for ${target.path}.`);
    return { filename, output, prepared };
  });
  const preparedCount = plans.filter(plan => plan.prepared).length;
  if (preparedCount !== 0 && preparedCount !== plans.length) {
    throw new Error('Incomplete GridSample patch: ESM and CJS differ. Reinstall with npm ci --ignore-scripts, then apply the patch.');
  }
  if (check && preparedCount === 0) {
    throw new Error('GridSample patch is not applied. Run node scripts/patch-jax-onnx-grid-sample.mjs.');
  }
  if (!check && preparedCount === 0) {
    for (const plan of plans) writeFileSync(plan.filename, plan.output);
  }
  return {
    package: packageName, version, patchId: 'grid-sample-unclipped-neighbors-v1',
    status: check ? 'verified' : preparedCount === 0 ? 'applied' : 'already-applied',
    licenseSha256,
    files: targets.map(({ namespace: _namespace, ...metadata }) => metadata),
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.slice(2).some(arg => arg !== '--check') || process.argv.length > 3) {
    throw new Error('Usage: node scripts/patch-jax-onnx-grid-sample.mjs [--check]');
  }
  console.log(JSON.stringify(applyGridSamplePatch({ check: process.argv.includes('--check') }), null, 2));
}
