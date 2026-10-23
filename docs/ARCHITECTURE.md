# Architecture (proposal; not implemented)

## Boundary
Cloudflare Pages serves static HTML/JS/WASM and public model assets. Browser Workers perform image preprocessing, training and inference; IndexedDB stores metadata and annotations; OPFS stores images and checkpoints; Cache API can cache versioned public assets. All browser storage is quota-limited and can be deleted: implement explicit project export/import and recovery.

Vite + React + TypeScript are frontend candidates. Initial target: desktop Chrome/Edge + WebGPU, subject to actual support and memory tests. GitHub Pages must support the /open-vision-builder/ base path and worker/model URLs; custom isolation headers cannot be assumed there.

## Learning engine
**Undecided.** Test jax-js and TensorFlow.js against real models, not just synthetic toy examples. ONNX Runtime Web is primarily for independent inference validation, not assumed to provide supported browser training. An ONNX inference graph is not equivalent to the original training graph. D-FINE needs model-specific losses, matching, gradient paths and handling of training-only outputs.

## ONNX-first artifact contract
Use a model-specific, versioned export recipe. Build a complete FP32 ONNX graph in the browser with updated and frozen weights, correct class-dependent shapes, preprocessing metadata, and output contracts. Compare trained-browser predictions against a fresh ONNX Runtime Web session, then against Python ONNX Runtime CPU in development CI. Python may prepare public base models and reference fixtures during development; users must never need a Python conversion step or an operator endpoint.

Export model.onnx plus labels, manifest (input layout/normalization, box format, postprocessing, opset, provenance), Python example, license notices and deployment kit. Keep optimizer/checkpoint state separate. Start with self-contained ONNX; consider external-data shards for static hosting where asset limits apply.

Cloudflare Pages has a per-asset size limit; check WASM and model assets at build time. Browser-generated local downloads are not subject to that static hosting limit.
