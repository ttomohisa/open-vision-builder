# Classification Real-Image Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Measure the existing classification app on pinned real photographs against the same native PyTorch recipe and independently verify its complete ONNX.

**Architecture:** Drive the existing App/Worker using Playwright, preserving its real encoded-image and download path. A test-only browser fixture records source features and final-head predictions; native timm calculates its own features from independently preprocessed PNGs. Separate browser context and Python ORT consume the exported complete model.

**Tech Stack:** Existing TypeScript/Vite/React, ORT Web, jax-js WASM, Python3.12, pinned timm/PyTorch/ORT/Pillow; no new dependency versions.

**Spec:** [Design and fixed conditions](../specs/2026-10-25-classification-real-images-design.md).

## Global constraints

- Existing MobileNetV4 Small checkpoint, source revision and licenses; only zero-initialized `[1280,5]` head and `[5]` bias train.
- 5 classes,6 train/2 unused tuning/4 evaluation per class,60 selected PNGs; App and experiment run only the50 train/evaluation images.
- 40 full-batch SGD updates, lr0.01, no momentum/decay/shuffle/augmentation; current production defaults.
- All tensor comparisons use original atol2e-4/rtol1e-4; normalized input maxAbs<1e-6. Quality criteria are defined in the design and JSON plan before model measurement.
- One completed experiment,10-minute runtime/90-minute initial active-work budget, no outcome-driven tuning. Preserve all failures; no detector rerun.
- Dedicated clean checkout/branch retains ignored dependency/model caches. Publish preregistration to GitHub before new real-image inference; user merges.

## Review focus

- A wrong class/split/order or duplicate author/photo/input must be rejected before training.
- Same exported model on both sides is not independent native learning; compare native source features and40 independently computed updates.
- Recorded loss40 precedes update40; final train metrics must use final head, not the last history element.
- Changed frozen model tensors or missing/mis-shaped classifier tensors must reject the export.
- A numerical/quality failure or interrupted run must leave an explicit failed/incomplete report, never overwrite previous evidence or silently retry.

## Tasks

### Task1: fixed data and provenance

Files: `tests/fixtures/classification-real-data.json`, `scripts/prepare_classification_data.py`, `tests/classification_data_test.py`, `docs/verification/2026-10-25-classification-data.md`.

Interface: `prepare(manifest_path, output_dir, archive_path=None)` verifies pinned archive/JPEG/PNG hashes, exact class/split/creator identities and regenerates only declared PNG files; output `images/{id}.png`, plus copied data manifest. No model computation. Manifest entries carry id, class index, split, original path/hash, canonical hash/bytes, dimensions, creator and source/license URL.

- [x] Write rejection tests for duplicate identities/split counts, path traversal, tampered image/manifest and unexpected output files; observe failure, implement, verify.
- [x] Record deterministic selection/exclusions and visual audit before measurements. Confirm exact source attribution.
- [ ] Commit data/provenance with the preregistration.

### Task2: independent native comparison and export verification

Files: `scripts/classification_real_reference.py`, `tests/classification_real_reference_test.py`.

Interface: CLI `--plan PATH --data PATH --images DIR --run DIR --source-model PATH --source-cache DIR`; reads saved `browser.json`, `fresh-browser.json`, `model.onnx`, `report.json`, `inputs/{id}.f32`; writes `native.json` (ignored arrays) and `comparison.json` (bounded summary/per-image results). Plan/data/model/code identities are validated. Native feature path is `forward_head(forward_features(x), pre_logits=True)`.

- [x] Test independent hand-computed classification metrics, wrong shapes/nonfinite arrays, exact loss timing and zero-head SGD fixture, frozen graph/head validation and an outside-tolerance image staying FAIL. Observe RED before implementation.
- [x] Implement pinned native loader and batch1 features,40 full-batch updates, final metrics, graph integrity and Python ORT comparisons. Save results before reporting a scientific FAIL; reject integrity errors with nonzero exit.
- [x] Run targeted tests and the Python suite. No large-model experiment until preregistration is published.

### Task3: browser/CLI orchestration

Files: `scripts/run_classification_real.mjs`, `scripts/classification-real-state.mjs`, `scripts/classification-worker-observer.mjs`, their matching declarations, `tests/classification-real-state.test.ts`, `tests/classification-worker-observer.test.ts`, `tests/fixtures/classification-real-browser.ts`, `tests/fixtures/classification-real-plan.json`.

Interfaces: state module validates fixed recipe/criteria and loss/summary/image identities; timeout returns positive integer milliseconds; refuses used output directories. Browser fixture exposes source-feature/input recording and isolated full-model inference using production decode/preprocess/predictHead. Controller drives real App file inputs, records actual Worker result/request metadata and ZIP, then closes training context before fresh inference.

- [x] Test changed criteria/steps, mismatched worker image/split metadata, nonfinite metrics, no-overwrite and timeout boundary. Observe RED then implement minimal behavior.
- [x] Build existing App and supplementary fixture. Run initial `--check` to validate bytes/bundle/browser import only, without model inference or training.
- [x] Complete the review fixes and rerun the full tests, typecheck, and updated no-inference Chromium preflight. Final build confirmation and GitHub CI remain in Task4.
- [ ] Inspect test-only GET routes and network records; no broad route access, no training inputs or source model in fresh-context verification.

### Task4: publish, run once, record and review

Files: result/evidence under `docs/verification`, AGENTS/README/STATUS/VERIFICATION/ROADMAP; CI adds only unit tests/preflight when useful, not automatic quality reruns.

- [ ] Publish exact data/plan/code as a draft PR before measurement; record SHA and original NOT RUN state.
- [ ] Execute one fixed comparison within cap. Save failures and per-image/confusion results, verify hashes, inspect misclassifications without selecting a new run.
- [ ] Independent whole-change code review and saved-result review; resolve material findings with meaningful regression tests.
- [ ] Run affected tests, typecheck/build, existing browser/ONNX regression gates via CI. Publish results and inspect latest-head CI, then tell user when ready to merge.

## Execution ledger

- PR #6 merged at 2026-10-25 10:12 JST, base `c088e211400e2db91e488858382fca39971cf41d`. Existing101 JavaScript tests passed before implementation.
- Ruling: continue the already authorized next task with existing UI defaults and current dependencies. No new product architecture approval is needed. Separate read-only code mapping and data audit inform bounded implementation tasks.
- Ruling: the current dedicated checkout is clean; use a new feature branch and keep its model/dependency caches rather than duplicate them in another worktree. No edits occur on main.
- Ruling: canonical lossless PNG inputs make independent decode/preprocess comparison useful without JPEG codec differences. Original JPEGs and conversion hashes remain attributable; JPEG generality is outside this experiment.
- New real-image model measurement: **NOT RUN**. Preregistration publication, its saved commit SHA, the one bounded measurement, result review, and GitHub CI remain pending. The JSON plan binds the dataset and current model artifact before execution.
- Task1: nine focused data-boundary tests observed RED then GREEN; official-archive preparation and an idempotent verification produced all 60 canonical PNGs. Dataset manifest SHA-256: `7d6885edda14c5e08b9a138d5f56ad9ab15b4d9ce3214b1d2befd9ceb5f80805`. Seven content-only exclusions are recorded. Both the data implementer and integrating agent visually inspected the final 60 images before model measurement; difficult photographs and photographic edits were retained.
- Source preparation: existing preparation passed its generated fixture. The regenerated ONNX is `eb823aa24c3d3308ae8e3d65074634aa96800831bb238114207315a79d540e8a` (15,069,036 bytes), also the currently deployed main Pages artifact, byte-identical on read-only download. The historical P1a artifact hash is `28857aef081dbba9e038c04cd8f2dabce28f654c66ffa47af1444179194f21d7`; that old binary is unavailable and the cause of the different bytes remains unknown. The JSON plan pins the actual current artifact before real-image measurements; the checkpoint revision/hash and recipe are unchanged.
- Task3 state boundaries: five focused tests observed RED then GREEN (fixed recipe, split/identity, actual Worker metadata/losses, output preservation, FP32 head orientation). Reuse the existing tested integer timeout helper instead of introducing another implementation.
- Verification after review fixes: observer gate tests 2/2, classification state tests 5/5, the full JavaScript suite 108/108, the full Python suite 112/112, typecheck and the final build passed. Updated no-inference Chromium preflight passed on Chromium 156.0.8078.4 with the expected plan/data/model hashes and 50 images. GitHub CI remains pending.
- Review outcomes: the actual Worker run request now passes the shared validator through `page.exposeFunction` before forwarding to the real Worker; a separate observer module and two focused tests cover that gate. The native inclusive 0.05 gap boundary was observed RED and corrected with only `8 * float64 epsilon` for arithmetic representation error. The 0.05 evaluation accuracy/macro-F1 criterion and all other numerical/quality thresholds remain unchanged. These are verification-code fixes, not evidence of a real-image model result.
- CI scope: reproduce the fixed classification inputs and run `node scripts/run_classification_real.mjs --check` after the existing public-image preflight. CI does not invoke this real-image learning/quality experiment. Its data manifest may be saved as evidence; input photographs stay out of Pages artifacts.
- Ruling: this preregistration covers canonical PNG, frozen MobileNetV4 Small features, classifier-head SGD, and WASM only. Broad JPEG decoder compatibility, WebGPU, detector qualification, and release readiness remain outside the result's scope.
