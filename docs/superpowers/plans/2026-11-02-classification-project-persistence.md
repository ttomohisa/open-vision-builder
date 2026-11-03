# Classification project persistence: implementation plan

Status: design checkpoint; **not implemented**. Based on merged PR #7. This is a deliberately bounded first step before touching the training path.

## Goal and boundary

Allow a user to save a classification dataset (original image bytes, class labels, train/evaluation assignments, and configuration), close the tab, and restore it without network services. Keep inference ONNX export separate from a restorable project. Do not claim that a project snapshot can resume a partially completed SGD run.

## First implementation slice

1. Inspect the actual App state, image decoding, dataset validation, and Worker job cancellation before coding. Reuse the existing validation and limits (2–8 classes, <=64 images, <=20 MiB/image, <=16 MP/image); never trust imported metadata.
2. Introduce a versioned, explicit manifest for classification projects with stable image identifiers, class identifiers, partition membership, original file MIME/size/hash, model recipe identifier, and training options. Define precise schema validation, duplicate IDs, unsupported versions, and file path traversal rejection. No executable payloads.
3. Add **explicit backup export/import first**. Package manifest and original image bytes in a ZIP, with SHA-256 per file and a manifest schema version. Import must verify all hashes and constraints before replacing current in-memory project; failure must leave current project intact. Never silently overwrite user work.
4. Add local autosave only after successful backup round-trip tests. Use IndexedDB for manifest and metadata, OPFS for large image bytes if available, and a transaction/commit-marker scheme that never exposes half-written projects. Define fallback for OPFS unavailable, quota errors, private browsing, and browser data deletion. Make storage status and persistence permission visible; do not call local storage a backup.
5. Keep project checkpoint and trained-model export distinct. Model ZIP currently exports a complete ONNX for inference, not optimizer state. A separate follow-up will define checkpoint versioning, optimizer state, update step, and deterministic training resume.
6. Keep static hosting, no backend/API/DB, no upload of user images. Avoid embedding user images or private test data in Git, CI artifacts, logs, or telemetry.

## Acceptance tests (before marking implementation complete)

- Save -> close -> reload -> restore: original bytes and image/class/split/settings match.
- Export -> import in fresh browser profile: manifest and image bytes match, including hash checks.
- Tampered hash, missing image, unsupported schema, duplicate ID, malformed archive, oversized decompression, and path traversal are rejected without modifying existing work.
- Interrupted write, quota failure, and cancellation do not expose partially saved projects.
- Changing images/labels invalidates stale training outputs; no old Worker response can replace restored state.
- Local persistence capability and backup instructions are accurate on GitHub Pages subpaths.
- Existing classification training, full ONNX export, Python inference, and detector G1 FAIL status remain unchanged.

## Execution order

A. Review existing state/ZIP/Worker code and document exact integration points.
B. Implement pure schema, ZIP serialization, validation and unit tests.
C. Wire explicit UI backup and restore with confirmation and browser integration tests.
D. Implement durable browser storage with failure recovery and tests.
E. Measure memory/storage and cancellation separately; propose training checkpoint/resume only after verifying optimizer-state contract.

## Non-goals

No server sync, no detector persistence, no WebGPU claims, no re-run of PR #7's one-shot real-image experiment, no relaxation of detector gates. Each implementation slice needs its own tested PR. User merges manually.

## PR #9 implementation checkpoint (2026-11-03)

The explicit ZIP-only slice now has source in `src/project-backup.ts`, UI integration in `src/App.tsx`, and unit/browser tests `tests/project-backup.test.ts` and `tests/project-backup.browser.spec.ts`. ZIP validation scans the central directory and verifies checksums before UI state mutation. Backup ZIP is not an ONNX model kit; no learned weights are saved. The current hard ZIP cap is 512 MiB to bound decompression and memory risk, even though the separate per-image limits could sum to more. **CI status and any failing checks must be captured before making this PR mergeable.** Follow-on IndexedDB/OPFS autosave and optimizer checkpointing remain outside this slice.
