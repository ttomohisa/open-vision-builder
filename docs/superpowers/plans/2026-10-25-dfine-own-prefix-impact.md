# D-FINE own-prefix impact implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development`; preserve the bounded experiment and its negative results.

**Goal:** Measure whether the existing browser final-head recipe learns differently from the same PyTorch recipe when each uses its own frozen-prefix features.

**Architecture:** A developer-only local controller supplies the same checksummed public image tensors to actual Chromium and PyTorch. Each computes its own features and independently matches targets. Short same-browser-feature controls separate head arithmetic from upstream differences; full browser-generated ONNX is reloaded in a fresh browser context and Python CPU.

**Tech stack:** Existing pinned D-FINE Nano COCO, Transformers/PyTorch CPU, ORT Web WASM, jax-js WASM, Playwright and ONNX Runtime Python. No production server, new backend, model replacement, or user-data upload.

**Spec:** [ADR 0005](../../decisions/0005-bound-numerical-investigation.md), [PRODUCT](../../PRODUCT.md), and [machine-readable preregistration](../../../tests/fixtures/detector-impact-plan.json).

## Scope and decisions fixed before model measurements

- Reuse the existing 80-class, eight-tensor pretrained final bbox/class-head recipe. Prefix and LQE remain fixed. No custom classes, auxiliary losses, all-layer training, or decoder replay.
- Use ten COCO val2017 images selected from annotation/license metadata without model predictions: four train, two reserved tuning, four evaluation. Preserve **every** instance annotation; reject any crowd image. The exact IDs, original bytes, annotations, attribution, splits, and canonical input hashes are frozen in the data manifest before inference.
- Evaluation means unused by **this adaptation experiment**. COCO validation data may have influenced the original checkpoint selection; this does not measure pretraining-unseen generalization or a custom task.
- Precompute one canonical input per public image: Pillow 11.1.0 RGB, bilinear stretch to 640×640, FP32 NCHW / 255. Both runtimes consume the same bytes. Browser decoding/resizing parity is outside this experiment.
- Fixed train order, three epochs, 12 batch-one SGD steps, learning rate 1e-4, no momentum/decay/augmentation/shuffle. Initial weights, targets and all other settings are identical. No tuning search; the two tuning images are descriptive only.
- At steps 0, 6 and 12, measure the first training image using current **browser** weights and browser features. Recompute forward, matching, loss, all gradients and a hypothetical SGD update in PyTorch. Use the existing pretrained-forward/loss/gradient/SGD tolerances unchanged. Do not advance the own-prefix reference trajectory during these controls. A failed control stops before another accepted update or evaluation.
- Save all raw per-step logits, boxes, matching, four loss terms, gradients and updated tensors under ignored artifacts. Commit compact comparisons and hashes. Never reorder queries or force native matching in the browser arm.
- Cache each runtime's own frozen features. Preserve prefix/model hashes, native frozen parameters/buffers, LQE bytes, and every unmodified ONNX initializer.

## Preregistered decision conditions

The numerical limits below are screening conditions for this **small experiment**, chosen before observing its predictions. They do not replace G1 or establish practical accuracy.

1. Both arms must reduce the mean training total loss by at least 1% over the fixed 12 steps. This distinguishes a visible learning change from an almost unchanged objective; it is not an overfitting criterion.
2. Each own-path per-step loss component and total must differ by at most 1% relative to PyTorch, denominator `max(abs(native), 1e-8)`. This screens the learning trajectory; all gradient/update differences and raw-output original-tolerance results are also reported.
3. Evaluation AP50 must differ between final arms by at most 0.01, and neither arm may lose more than 0.02 AP50 relative to its initial model. Compute class-aware score-ranked matching at IoU 0.5, threshold 0.05, global top 100 query/class predictions per image, no NMS, stable ties, 101 recall points, averaging classes with evaluation ground truth. Record TP/FP/FN including predictions for absent classes. This tiny slice cannot establish population quality.
4. The complete browser-exported FP32 ONNX must agree with its final browser head outputs in a **fresh browser context** and Python CPU using the existing final logits/boxes tolerances. The fresh context receives only model bytes and canonical inputs, not training features or weights.
5. Controls must pass; values must be finite; pretrained localization and classification must actually update; all frozen tensors must remain unchanged.

| Result | Next action |
| --- | --- |
| Conditions hold | Keep the candidate for the next bounded custom-class/training-scope work; retain old G1 FAIL. |
| Same-feature control fails | Save and stop. Investigate the failed head/loss/update comparison only; do not resume decoder decomposition. |
| Controls pass but own-path conditions fail | Use the actual failing input, matching, loss or update to choose one further action. No threshold fitting or automatic operator sweep. |
| PyTorch does not improve, budget is exceeded, or evidence is incomplete | Mark inconclusive and reassess this recipe/data fit. Do not blame the runtime or replace the model without evidence. |

One completed experiment, no automatic retries or hyperparameter changes, 15-minute runtime limit and 90-minute initial active-work budget including implementation and recording. Infrastructure failures are recorded; no additional measured run is started to seek a better outcome. Stop and preserve a reviewable checkpoint if the budget is exhausted.

## Global constraints

- All models/data/raw tensors stay outside `public/` and `dist/`; no detector UI or Pages publication.
- Developer localhost GET routes serve fixed public fixtures only. Browser data are read by the local test controller, not posted to an application API.
- Dataset/plan hashes must be bound before measurements. A missing manifest digest must fail closed.
- User merges the PR. PR #5 merged as `f73cdb9407f60fb402e23afa9abdefdd7e98e058`; this work is on `test/dfine-own-prefix-impact`.

## Review focus

- Incorrect COCO class mapping, ignored other objects, duplicate images across splits, or altered bytes must be rejected by data tests.
- Nonfinite/missing/reshaped tensors and same-feature mismatch must fail before advancing the browser experiment.
- A control must use the exact current browser features/weights and must not update the independent native trajectory.
- Duplicate detections and wrong-class boxes must not inflate AP or recall; use hand-computed metric tests.
- A failed/aborted run must preserve its completed prefix and must never leave a stale completed report.

## Tasks and verification

- [ ] **Data:** `scripts/prepare_detector_impact_data.py`, `tests/detector_impact_data_test.py`, data manifest and provenance note. Write meaningful rejection tests, observe failure, implement, rerun. Pin the manifest hash in the plan before any model measurement.
- [ ] **Reference/control:** `scripts/detector_impact_reference.py`, `tests/detector_impact_reference_test.py`. Reuse the pinned source loader, wrapper and head/loss functions; test tensor contracts, independent controls and hand-derived metrics before implementation. Keep raw evidence ignored.
- [ ] **Browser/controller:** `tests/fixtures/detection-impact-browser.ts`, `scripts/run_detector_impact.mjs`, focused controller-state tests. Exercise actual browser prefix/head and existing SGD/export code. Persist each checkpoint and stop on failed controls. Verify fresh-context and Python full-model inference.
- [ ] **Run/record:** Publish the preregistration/code checkpoint before measurements. Execute the one fixed experiment, preserve its actual outcome, update STATUS/VERIFICATION and this ledger. Run affected suites and build; obtain independent review; push the final evidence and inspect CI before asking for merge.

## Execution ledger

- 2026-10-25 JST: PR #5 merge verified through GitHub and fetched. Clean dedicated branch from the merge commit. Existing Python suite: 53 passed. No new model measurements yet.
- Implementation choice: canonical FP32 image tensors isolate prefix/learning differences from decoder/resize differences; this reduces scope and does not validate the product image loader.
- Implementation choice: use the existing one-step browser training API even though recording gradients repeats one computation; avoid changing production training mathematics for a developer experiment.
- Pre-measurement checkpoint: `f5b07fc82d950106c2e4165fea3869236c488b2f`, PR #6. Python 83 / JavaScript 100 tests, typecheck, build and actual Chromium 156.0.8078.4 module-import preflight passed. Independent review found no Critical/Important issue; exact-criteria validation was strengthened with observed failing regressions before this checkpoint.
- Infrastructure ruling: the first attempt stopped at checkpoint 0 before Python control execution because Node requires an integer child-process timeout, while the monotonic remaining time was fractional. Accepted updates: 0; completed controls: 0. Preserve `artifacts/p1d/run/` and its [failure record](../../verification/evidence/2026-10-25-impact-infrastructure.json). Add an integer-timeout regression and fix only this controller boundary. A repaired attempt uses a new output directory; no image, split, training parameter, comparison limit, or completed scientific result changes. This is one explicit infrastructure repair, not an automatic search for a passing outcome; the original active-work budget continues.
