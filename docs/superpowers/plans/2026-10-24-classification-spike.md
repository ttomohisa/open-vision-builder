# Browser classification and ONNX implementation plan

> **For agentic workers:** Use the approved PRODUCT, ARCHITECTURE and VERIFICATION documents. Execute the tasks and record evidence; user review occurs on the implementation PR. Do not merge it.

**Goal:** Deliver the first usable P1 slice: local image classification transfer learning on a frozen, pretrained MobileNetV4 Conv Small, a complete browser-generated ONNX, and reproducible browser/Python checks.

**Architecture:** ONNX Runtime extracts frozen 1280-dimensional features in a Worker. jax-js computes softmax cross-entropy gradients and SGD updates for a new class head. A model-specific protobuf exporter prunes the original classifier and joins the learned head to the complete frozen graph. Developer Python prepares public assets; user training/export never invokes Python or an API.

**Tech Stack:** React 19.3.1, TypeScript 7.0.2, Vite 8.3.5, ONNX Runtime Web 1.30.0, jax-js 0.1.27, optax 0.1.2, onnx-buf 1.20.0-1, protobuf 2.16.0. Exact dependencies and transitive versions go in package-lock.json. Developer reference: Python 3.12, CPU torch 2.6.0, torchvision 0.21.0, timm 1.0.15, ONNX 1.17.0, ONNX Runtime 1.20.1.

**Spec:** docs/PRODUCT.md, docs/ARCHITECTURE.md, docs/MODELS.md, docs/VERIFICATION.md, ADR 0001.

## Global constraints and scope

- No operator account service, API, DB, training endpoint, user-data upload or telemetry.
- All user image processing, head training, evaluation and full ONNX export run locally in the browser.
- Preserve the pretrained frozen feature extractor in the downloaded model. Head-only artifacts do not satisfy export.
- Classification requires at least two classes and a disjoint training/evaluation split. One-class softmax is rejected instead of pretending its zero gradient is training.
- P1 remains incomplete: detector fine-tuning, backbone fine-tuning, real-world quality qualification, hardware memory measurements and resume/backup qualification remain open.
- Published jax-js GridSample has a reproducible boundary discrepancy relevant to D-FINE's default sampling method. Record the probe and block detector support until corrected and revalidated.
- Synthetic generated images establish pipeline correctness only; their accuracy is not a real-world benchmark.
- GitHub Pages initialization may require repository settings. Prepare and test the deployment workflow; never claim an unpublished URL works.

## Review focus

- Class edits, image edits and cancelled/repeated jobs must invalidate old results and prevent stale ONNX downloads.
- Empty, one-class, non-finite or shape-inconsistent data must fail before changing a model.
- Export must preserve every needed frozen tensor, embed updated head tensors, remove stale classifier outputs and require only the image input.
- Sample and user files must remain local; runtime and model requests must be static GET requests.
- Vite subpaths, Worker imports and WASM assets must work without cross-origin isolation.

## Task 1: Reproducible public model preparation

**Files:** scripts/prepare_model.py, scripts/requirements-models.txt, scripts/predict.py; generated public/models/mobilenetv4-small/* (ignored by git).

**Contract:** model.json schemaVersion=1 contains source model/revision/hash, artifact file/hash/bytes, input name/shape/normalization/resize, features name/size, classifier mapping, reference file and licenses. Full backbone.onnx has images [1,3,224,224] input and logits [1,1000] plus features [1,1280] outputs. reference.json includes deterministic fixture recipe and expected native logits/features. Prepared assets contain required license notices.

- [x] Pin source safetensors at revision 331fb803779522b685cf942e15f914fb6741c1eb; verify SHA-256 7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4 before loading.
- [x] Load without pickle; export the complete eval model with explicit features using opset 17, then check ONNX structure.
- [x] Compare native PyTorch and Python ONNX Runtime for the same deterministic input; predeclared FP32 atol=2e-4, rtol=1e-4. Fail preparation on mismatch, unexpected shape or external data.
- [x] Write manifest, reference fixture, ONNX hash and source/Apache notices. No generated binary is committed.
- [x] Provide Python inference/parity helper that accepts an exported full ONNX and a report fixture.

## Task 2: Real head training and complete ONNX export

**Files:** src/ml/head.ts, src/ml/onnx.ts, src/ml/preprocess.ts, src/ml/types.ts; tests/head.test.ts, tests/onnx.test.ts, tests/preprocess.test.ts; scripts/probe-dfine.mjs and docs/verification/2026-10-24-dfine-probe.md.

**Interfaces:** HeadWeights { weights: Float32Array, bias: Float32Array, featureSize: number, classes: number }; trainHead(features, labels, options) returns weights and history with progress/cancel callbacks. predictHead(features, weights) returns logits. exportClassifier(modelBytes, manifest, weights, labels) returns Uint8Array of complete ONNX. preprocessRgba(rgba,width,height,size) returns normalized NCHW Float32Array.

- [x] Before implementing training, test the independently derived two-feature/two-class gradient, actual weight changes and loss reduction; reject one class, invalid labels, NaN and shape errors.
- [x] Implement FP32 jax-js valueAndGrad and fixed-rate SGD, explicit tensor ownership/disposal and yielding between epochs; preserve only completed steps on cancellation.
- [x] Before implementing export, test a full tiny known backbone graph joined to a changed class head in a real runtime. Check that updated class logits, all required frozen weights and one image input survive serialization.
- [x] Implement graph ancestor pruning and MatMul/Add head with explicit shapes, finite checks and embedded FP32 tensors; reject unsupported manifests/external weights.
- [x] Test pure bilinear half-pixel resizing and ImageNet normalization against hand-derived pixel fixtures. Use the same formula in Python reference processing.
- [x] Preserve the known D-FINE GridSample discrepancy in a runnable, separately reported probe. Do not turn a failed detector gate into a passing application test.

## Task 3: Browser workflow, independent verification and deployment

**Files:** package.json/package-lock.json, tsconfig.json, vite.config.ts, index.html, src/App.tsx, src/main.tsx, src/styles.css, src/worker.ts, src/protocol.ts, scripts/copy-runtime.mjs, tests/browser.spec.ts, .github/workflows/ci.yml, .github/workflows/pages.yml, docs/DEPLOYMENT.md, docs/STATUS.md and evidence record.

**Contract:** worker requests carry job IDs. A job contains the immutable class list and Blob samples with train/test labels. Progress, result and error responses echo the ID. Model import/extraction/training/evaluation/export are Worker tasks. User edits cancel/invalidate old results. UI clearly identifies this as the classification P1 slice.

- [x] Add pinned build/test dependencies and a versioned asset-copy step; no app dependency is downloaded from a CDN at runtime.
- [x] Implement local image/class input, explicit train/test groups, deterministic sample images, progress/cancel, held-out metrics and model/labels/manifest/report downloads. Limit batch size and validate input before sending work.
- [x] Run actual Chromium through the complete workflow with real prepared MobileNet weights. Verify native forward fixture tolerance, weight changes, trained-browser vs fresh ONNX session tolerance, download and cancellation/stale state behavior. Use a nested base path without isolation headers.
- [x] Read the browser-generated ONNX in Python CPU and compare the exact saved input/logits with atol=2e-4, rtol=1e-4; record measured results rather than assuming parity.
- [x] Add CI build/typecheck/unit/browser/reference checks and static build artifacts. Add Pages deployment only for the trusted same-repository implementation branch and main. Keep app build permissions separate from deployment permissions.
- [x] Update STATUS, README and ADR with the implemented subset, verified environments, failed detector gate and remaining work; complete independent review and prepare the PR description for main. Do not merge.

## Execution record

- 2026-10-24: PR #2 merged at 64e9fe53579ca95d464ed85944f9bd3ef237336e. Isolated clone and feature branch created. Existing repository has no application or test baseline.
- Decision: classification uses ORT frozen features plus jax-js head training. This is a bounded transfer-learning proof, not proof that arbitrary ONNX or D-FINE can backpropagate through jax-js. The complete P1 gates remain unchanged.

- Whole-branch review identified a file-decoding parity defect: 8-bit Canvas changed partially transparent RGB before float compositing. Fix uses native ImageDecoder pixel copy and explicit EXIF orientation; file-to-tensor browser regressions are required before final verification. Minor PR metadata and sample replacement notices were corrected.
- Static application notices are collected for the actual Worker bundle, including generated ONNX protobuf and bundled third-party portions; model notices stay in the deployment ZIP.

- Final local verification: build/typecheck PASS; JS36/Python11/browser27 PASS; fresh browser export vs PythonCPU maxAbs1.0132789611816406e-6 under unchanged tolerances. Independent reviewer reports no remaining Critical/Important findings. Complete head model SHA78b7e3d6a5c6c186587cd66c80e7c971c10a4d11c45088d2f857ac921db30412.
- The implementation PR is the handoff for user review and merge. Pages initial settings still require an authenticated owner operation; workflow and exact steps are prepared. No live preview success is claimed.
