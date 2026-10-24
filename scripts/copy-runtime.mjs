import { copyFile, mkdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { collectLicenses } from './collect-licenses.mjs';

await collectLicenses();
const target = 'public/runtime/onnxruntime';
await mkdir(target, { recursive: true });
for (const file of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  const source = join('node_modules/onnxruntime-web/dist', file);
  if ((await stat(source)).size > 25 * 1024 * 1024) throw new Error(`${file} exceeds the static asset limit`);
  await copyFile(source, join(target, file));
}
await copyFile('node_modules/onnxruntime-web/README.md', join(target, 'README.md'));
await copyFile('third_party/onnxruntime/LICENSE', join(target, 'LICENSE'));
await copyFile('third_party/onnxruntime/ThirdPartyNotices.txt', join(target, 'ThirdPartyNotices.txt'));
console.log('Copied the pinned single-thread WASM runtime for same-origin loading.');
