# Classification draft autosave: bounded implementation and verification

Date: 2026-11-03 JST. Based on merged [PR #9](https://github.com/ttomohisa/open-vision-builder/pull/9).

## Goal

After editing a classification dataset and waiting for the browser to confirm a successful autosave, users can close/reopen the tab and recover the original image bytes, labels, train/test split, and training settings. This does **not** restore completed inference ONNX, optimizer parameters, a partially trained head, or an unfinished training job. Portable ZIP backup remains independent.

## Design decision for this slice

Use one IndexedDB database (`open-vision-builder-classification-draft-v1`) and one record (`drafts/current`) for metadata and original image Blobs. A `readwrite` transaction using the `strict` durability hint writes the *entire* snapshot at once. A failed/aborted/quota-exceeded transaction must leave the last committed record intact. This avoids a two-storage commit protocol that would otherwise be needed with both IndexedDB and OPFS. **OPFS and large-image performance measurements are intentionally deferred**; do not claim this slice implemented OPFS.

- Store a versioned snapshot with source Blob, SHA-256, ID, name, label, split, synthetic flag, raw epoch and learning-rate strings. Invalid in-progress edits (empty label, malformed numeric input) are valid drafts, even though the training path rejects them.
- Debounce changes for 700ms; serialize requests within the tab. Only update the saved indicator on confirmed IndexedDB transaction completion. Unchanged hydration/run completion must not create a new revision.
- Transaction compares a monotonically increasing revision against the caller's last-read revision. A second stale tab fails closed and shows recovery instructions rather than overwriting the newer draft.
- Before applying a restored snapshot, validate structure, image hash, MIME, size, and actual decodability/16MP limit. Failure blocks autosave and leaves the stored record intact; portable ZIP restore is an independent recovery route.
- Read availability, quota estimate and persistence grant from Storage API. A user-initiated `persist()` request reports the actual result rather than promising a grant. Origin storage may still be deleted by the user; neither IndexedDB nor persistent storage replaces an external ZIP backup.
- During a training job or manual ZIP import, postpone autosave; completion writes the current dataset. Existing job ID/Worker cancellation remains the authority for invalidating trained results when an import replaces the data.

## Acceptance tests

1. Sample images/labels/settings -> wait for confirmed save -> reload, verify bytes/labels/split/settings. Include a partial/invalid in-progress name.
2. Two tabs open the same revision, first writes, second must not overwrite and must show a conflict; a subsequent reload recovers first tab's version.
3. Abort a write after staging it inside the IndexedDB transaction; reload must recover the previous completed draft.
4. Tamper an image SHA-256 in IndexedDB; reload must reject the record instead of silently replacing it with the empty initial state.
5. Current project ZIP backup/restore remains independent, with no new external HTTP writes.
6. Original classification training, full ONNX export, independent Python parity, detector qualification status remain unchanged.

## Out of scope

OPFS copy-on-write staging and garbage collection, multi-project catalog, cross-device backup sync, guaranteed retention after site-data clear/eviction, training checkpoint and optimizer resume, detector data persistence. No claim that debounced writes can finish after the tab is force-closed; the UI must tell users when a save is confirmed.

## Test record

Pending latest-head CI verification. Do not mark as complete or mergeable solely because code is committed.
