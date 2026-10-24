# D-FINE Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Correct GridSample and establish a bounded, real pretrained D-FINE Nano head-training and complete-ONNX verification path.

**Architecture:** Keep the static-only application and classify/detect qualification boundaries. Patch only verified jax-js ONNX bundles; prepare the pinned pretrained detector with developer Python; update the final pretrained detection head in browser-compatible TypeScript and export all frozen plus trained weights.

**Tech Stack:** Existing Vite / TypeScript / ONNX Runtime Web / jax-js, PyTorch 2.6 CPU and Transformers 4.56.0 for independent developer references.

**Spec:** [D-FINE foundation design](../specs/2026-10-24-dfine-foundation-design.md).

## Global Constraints

- No operator API / DB / training service; no user data leaves the browser.
- Keep complete FP32 ONNX mandatory; no Python conversion for exported user models.
- Use verified COCO-only Nano safetensors revision `066438d3d8f0da137a37b38fdf3368fd4afceced`.
- Never call bounded 80-class final-head training the complete native detector training recipe or independent custom-class quality qualification.
- Preserve predeclared tolerances, exact provenance, LICENSE / NOTICE, and 25 MiB static-asset limits.
- Keep binaries / large tensors / local virtual environments out of git. User merges the PR.

## Review Focus

- Both ESM and CJS imports must receive the correction; an unknown or partially modified package must fail before writes.
- At image borders, both feature and coordinate gradients must follow the same four neighbors as forward.
- The exported pretrained head may use aliases/transposes; a name-only replacement must not silently update the wrong parameter.
- Empty targets and non-overlapping or tiny boxes must produce finite, defined losses without differentiating matching.
- A complete model must work in a new runtime without borrowed tensors, feature cache, or a source checkpoint.

## Tasks

### 1. Repair and independently qualify GridSample

Files: `scripts/patch-jax-onnx-grid-sample.mjs`, `scripts/probe-dfine.mjs`, Python reference generator, GridSample and patch regression tests, provenance notes.

Interface: explicit patch apply / `--check`; existing `probe:detector` keeps the original six fixture expectations and reports genuine pass/fail.

- [x] Reproduce the upstream failure on merged main.
- [x] Run new independent forward/VJP tests against unmodified bundles and observe the actual boundary failure.
- [x] Implement full-source-hash-guarded four-line ESM/CJS correction; test drift rejection and idempotence.
- [x] Compare PyTorch, patched ONNXModel, original fixed fixtures, and a browser execution of the corrected operator.

### 2. Prepare the exact complete detector and head mapping

Files: `scripts/prepare_detector.py`, `scripts/requirements-detector.txt`, `tests/detector_model_scripts_test.py`, ignored `artifacts/p1b/detector-assets/` (no product distribution).

Interface: complete inference ONNX, frozen-prefix outputs, exact source/shape/initializer mapping, initial head parameters, procedural public fixture, source notices and a versioned manifest.

- [x] Pin implementation, weights, configuration and notices; verify safe loading and the historical Nano conversion fix.
- [x] Add rejection tests for wrong hashes/configuration/parameter mappings before implementing preparation.
- [x] Export and compare three generated fixtures in PyTorch and Python ORT at the spec tolerances.
- [x] Inventory graph operators and validate all final-head aliases against actual source tensors.

### 3. Implement final-head learning against an independent reference

Files: `src/ml/detection/head.ts`, `loss.ts`, `matching.ts`, tests, independent Python head reference.

Interface: typed frozen-prefix features and exact pretrained head parameters; forward logits/boxes; no-grad assignment; finite loss/gradient/SGD update and serialized final tensors for the exporter.

- [x] Establish forward, loss, assignment and one-step parameter-update fixtures before implementing TypeScript.
- [x] Implement final pretrained bbox MLP / class projection and frozen LQE / integral / box decoding.
- [x] Verify VFL / L1 / GIoU, assignment and gradients, including empty targets and invalid boxes.
- [x] Update actual pretrained localization parameters and verify changed predictions plus fixed-prefix invariance.

### 4. Export and run the browser path

Files: `src/ml/detection/export.ts`, focused exporter tests, browser verification harness, independent Python detector verifier, package/CI wiring as needed.

Interface: full prepared ONNX + exact mapping + trained tensors → a standalone ONNX; report has recipe/scope/source hash, parameter update evidence and browser/Python comparisons.

- [x] Write a meaningful full-graph replacement regression that fails without parameter substitution and rejects unknown/invalid tensors.
- [x] Implement exact initializer replacement and remove only explicit training-only outputs.
- [x] Run actual browser pretrained-head updates and fresh-session inference; compare the exported artifact in Python CPU.
- [x] Preserve classification regression tests and static-subpath behavior; record actual resource/environment limits.

### 5. Record, review and submit

Files: `README.md`, `AGENTS.md`, `docs/STATUS.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `MODELS.md`, `VERIFICATION.md`, ADR 0003 and dated evidence.

- [x] Record PR #3 merge and successful Pages live verification in repository docs.
- [x] Record this PR's measured results, source decisions, failures and remaining full-model/custom-class/quality gates.
- [x] Run relevant unit, Python, build and browser verification with clean dependency preparation.
- [ ] Obtain an independent whole-branch review, resolve concrete findings, create a PR, and verify remote CI. Do not merge.

## Recorded outcome and scope amendment

The unchanged native-prefix gate fails on intermediate features; completing the comparison task is not a numerical PASS. ADR 0003 records the supported strict nonzero qualifier, failed evidence preservation, component regression checks, explicit FAIL in browser/kit/CI, and exclusion of detector assets from public/Pages. Forward/head-gradient/update/export component results are recorded in `docs/verification/2026-10-24-dfine-foundation.md`. Full detector training, custom classes, quality and resources remain pending.
