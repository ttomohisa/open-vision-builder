import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { applyGridSamplePatch } from '../scripts/patch-jax-onnx-grid-sample.mjs';

const temporaryDirectories: string[] = [];
const files = ['dist/index.js', 'dist/index.cjs'];

function upstreamCopy() {
  const directory = mkdtempSync(join(tmpdir(), 'ovb-grid-sample-'));
  temporaryDirectories.push(directory);
  cpSync('node_modules/@jax-js/onnx', directory, { recursive: true });
  // Reconstruct the exact published source if the working install is already
  // prepared. The installer's whole-file hashes verify this test input too.
  for (const [filename, namespace] of [['dist/index.js', 'numpy'], ['dist/index.cjs', '_jax_js_jax.numpy']]) {
    const target = join(directory, filename);
    let source = readFileSync(target, 'utf8');
    for (const [axis, size] of [['x', 'Win'], ['y', 'Hin']]) {
      source = source.replace(
        `${namespace}.clip(${axis}0f.ref, 0, ${size} - 1)`,
        `${namespace}.clip(${axis}0f, 0, ${size} - 1)`,
      ).replace(
        `${namespace}.astype(${namespace}.clip(${namespace}.add(${axis}0f, 1), 0, ${size} - 1), ${namespace}.int32)`,
        `${namespace}.clip(${namespace}.add(${axis}0c.ref, 1), 0, ${size} - 1)`,
      );
    }
    writeFileSync(target, source);
  }
  return directory;
}

function contents(directory: string) {
  return files.map(file => readFileSync(join(directory, file), 'utf8'));
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe('explicit GridSample dependency patch', () => {
  it('check rejects an unprepared clean install without writing either bundle', () => {
    const packageDirectory = upstreamCopy();
    const before = contents(packageDirectory);
    expect(() => applyGridSamplePatch({ packageDirectory, check: true })).toThrow(/not applied/);
    expect(contents(packageDirectory)).toEqual(before);
  });

  it('prepares both published formats, is idempotent, and passes read-only check', () => {
    const packageDirectory = upstreamCopy();
    const before = contents(packageDirectory);
    expect(applyGridSamplePatch({ packageDirectory }).status).toBe('applied');
    const prepared = contents(packageDirectory);
    expect(prepared[0]).not.toEqual(before[0]);
    expect(prepared[1]).not.toEqual(before[1]);
    expect(applyGridSamplePatch({ packageDirectory }).status).toBe('already-applied');
    expect(applyGridSamplePatch({ packageDirectory, check: true }).status).toBe('verified');
    expect(contents(packageDirectory)).toEqual(prepared);
  });

  it('rejects CJS source drift before writing the valid ESM bundle', () => {
    const packageDirectory = upstreamCopy();
    const target = join(packageDirectory, files[1]);
    writeFileSync(target, `${readFileSync(target, 'utf8')}\n// unexpected source\n`);
    const before = contents(packageDirectory);
    expect(() => applyGridSamplePatch({ packageDirectory })).toThrow(/unrecognized.*SHA-256/i);
    expect(contents(packageDirectory)).toEqual(before);
  });

  it('rejects an incomplete patch instead of silently accepting mixed runtimes', () => {
    const packageDirectory = upstreamCopy();
    const originalCjs = readFileSync(join(packageDirectory, files[1]), 'utf8');
    applyGridSamplePatch({ packageDirectory });
    writeFileSync(join(packageDirectory, files[1]), originalCjs);
    const before = contents(packageDirectory);
    expect(() => applyGridSamplePatch({ packageDirectory })).toThrow(/incomplete/i);
    expect(() => applyGridSamplePatch({ packageDirectory, check: true })).toThrow(/incomplete/i);
    expect(contents(packageDirectory)).toEqual(before);
  });

  it('rejects a different package version without modifying source', () => {
    const packageDirectory = upstreamCopy();
    const manifestPath = join(packageDirectory, 'package.json');
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    writeFileSync(manifestPath, JSON.stringify({ ...manifest, version: '0.1.3' }));
    const before = contents(packageDirectory);
    expect(() => applyGridSamplePatch({ packageDirectory })).toThrow(/expected @jax-js\/onnx@0\.1\.2/i);
    expect(contents(packageDirectory)).toEqual(before);
  });

  it('rejects changed upstream attribution before modifying source', () => {
    const packageDirectory = upstreamCopy();
    writeFileSync(join(packageDirectory, 'LICENSE'), 'modified license');
    const before = contents(packageDirectory);
    expect(() => applyGridSamplePatch({ packageDirectory })).toThrow(/license/i);
    expect(contents(packageDirectory)).toEqual(before);
  });
});
