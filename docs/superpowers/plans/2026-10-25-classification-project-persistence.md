# Classification Project Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist and restore the current classification project locally and provide a validated portable backup without any server.

**Architecture:** IndexedDB stores the versioned project manifest while OPFS stores image blobs and the optional trained model-kit ZIP. A focused project module validates and serializes state; App.tsx hydrates once before enabling edits, debounces autosaves, and exposes backup/import/clear/persistence controls.

**Tech Stack:** React 19, TypeScript, IndexedDB, OPFS, Web Crypto SHA-256, fflate 0.8.2, Vitest, Playwright Chromium.

**Spec:** docs/superpowers/specs/2026-10-25-classification-project-persistence-design.md

## Global Constraints
- No operator API/database/accounts or user-data upload.
- Classification-only and one local default project in this PR.
- Images remain capped at 64 and 20 MiB each; backup parser enforces limits before replacement.
- Backup schema version 1 and every binary entry has byte length + SHA-256.
- Restore/import fails closed and never replaces valid current state with partial data.
- Training checkpoint/optimizer resume is out of scope.

## Review Focus
- Reload must not overwrite a valid project with the empty initial state.
- Path traversal and duplicate ZIP entries must be rejected before writes.
- Quota exhaustion must leave in-memory work usable.
- Object URLs must be revoked when samples are replaced/cleared.
- Restored trained result remains downloadable and edits invalidate it.

### Task 1: Project format and backup codec

**Files:** create src/project/types.ts, src/project/backup.ts; test tests/project-backup.test.ts.

- [ ] Write failing round-trip tests with images and optional model kit.
- [ ] Add bad schema/task/model, path traversal, checksum/size, label/split/settings and limits failures.
- [ ] Run focused Vitest and observe RED.
- [ ] Implement validation, SHA-256 and fflate ZIP codec.
- [ ] Run focused Vitest and observe GREEN.

### Task 2: IndexedDB + OPFS local store

**Files:** create src/project/storage.ts; test tests/project-storage.browser.spec.ts.

- [ ] Write browser save/reload byte-identity test.
- [ ] Add missing-file corruption and clear cases.
- [ ] Observe RED, implement manifest-last IDB + OPFS store, observe GREEN.

### Task 3: App hydration, autosave and backup UI

**Files:** modify src/App.tsx, src/styles.css, tests/browser.spec.ts.

- [ ] Add reload autosave restoration test.
- [ ] Add trained-result reload/download test.
- [ ] Add backup export → clear → import test.
- [ ] Add corrupted-backup leaves current project unchanged test.
- [ ] Observe RED, implement hydration/debounced autosave/result restore/UI, observe GREEN.

### Task 4: Documentation and full regression

**Files:** README.md, docs/ARCHITECTURE.md, docs/STATUS.md, docs/ROADMAP.md, AGENTS.md.

- [ ] Record exact persistence boundary and unsupported checkpoint resume.
- [ ] Run full JS/Python/build/browser verification in CI.
- [ ] Request independent code review and fix Critical/Important findings with regression tests.
- [ ] Verify latest branch CI and prepare PR for owner merge.
