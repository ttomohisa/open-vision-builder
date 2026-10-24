# Static application dependency notices

`npm run copy:runtime` copies the pinned ONNX Runtime assets and generates
`public/runtime/THIRD_PARTY_LICENSES.txt`. Vite copies it unchanged into
`dist/runtime/THIRD_PARTY_LICENSES.txt`; deploy it alongside the application.
The files under `public/runtime` and `dist` are generated and ignored by Git.

The collector is offline: it reads the installed npm LICENSE files and these
checked-in upstream notices. It checks the installed and locked package versions,
records each npm tarball URL/integrity, and includes the SHA-256 of every notice.
Missing/empty files or changed package versions fail the copy/build step.
No license is selected here for Open Vision Builder itself or model weights.

## Audited scope

On 2026-10-24, Vite's `generateBundle` module inventory was inspected separately
for the main build and the Worker build with `build.write: false`:

| Distributed code | Included package notices |
| --- | --- |
| Main UI | `react@19.3.1`, `react-dom@19.3.1`, `scheduler@0.28.0` |
| Vite modulepreload helper | Core MIT notice from `vite@8.3.5`; build-tool-only dependency notices are excluded |
| ML Worker | `@jax-js/jax@0.1.27`, `@jax-js/optax@0.1.2`, `fflate@0.8.2`, `onnx-buf@1.20.0-1`, `@bufbuild/protobuf@2.16.0`, `onnxruntime-web@1.30.0` |
| Code bundled inside ORT Web | `onnxruntime-common@1.30.0`; confirmed in `ort.wasm.bundle.min.mjs.map` |
| ORT native/WASM dependencies | Complete upstream `ThirdPartyNotices.txt` from the same ORT revision |
| Other source notices | ONNX schema Apache-2.0; protobuf varint BSD; jax-js references to tinygrad and TensorFlow.js |

The ORT `wasm` entry does not bundle the npm JavaScript packages `flatbuffers`,
`guid-typescript`, `long`, `platform`, or `protobufjs` and its `@protobufjs/*`
dependencies. Those are installed for other ORT backends. `@jax-js/onnx` is used by
the developer detector probe, not imported into the static application. They are
not represented here as distributed packages. The full upstream ORT notice remains
included for the native dependencies in the WASM binary.

Vite's separate `dist/THIRD_PARTY_LICENSES.md` currently reports the main UI
dependencies only. It is not the complete notice for the Worker and copied WASM.

## Updating

When a dependency version, runtime import, or ORT entry point changes, inspect
both bundle inventories again, including upstream source maps for code already
bundled by a dependency. Update the version/file inventory in
`scripts/collect-licenses.mjs` and the corresponding exact upstream sources and
hashes here. Preserve original notices and do not infer a license from the package
name alone; `onnx-buf` has different npm metadata and generated-file notices.

Run `npm run copy:runtime`, then the application build. Confirm the aggregate and
the ORT notices exist under `dist/runtime`, and that the aggregate contains each
package in the inventory. The copy step does not download licenses during builds.
