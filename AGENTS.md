# Open Vision Builder — Agent instructions

Before every change read README.md, docs/PRODUCT.md, docs/STATUS.md, docs/ROADMAP.md and the relevant architecture/model/decision documents. Inspect current code and PRs rather than trusting documents alone.

## Non-negotiable constraints
- This is independent of Browser Kitty.
- Deploy as a static application (Cloudflare Pages); no operator-run API, database, training server, or user-data upload.
- Annotation, fine-tuning, evaluation, checkpoints and ONNX generation run on the user's own browser and compute.
- Initial release covers image classification AND object detection, including real pretrained localization fine-tuning, not merely crop classification or inference.
- Export a complete, self-contained FP32 ONNX model containing frozen and updated weights. The user must not need Python to convert it.
- Supply Python/ONNX Runtime examples and self-hosting kits; users host their own inference.
- Check commercial redistribution rights of exact pretrained weight files, not just code licenses.
- Submit work as reviewable PRs. The repository owner merges; do not merge or push to main.

## Evidence and documentation
Never claim browser training, ONNX export, accuracy, runtime or commercial clearance without test evidence. Distinguish requirements, candidates and verified implementation. Before changing architecture or models, read docs/decisions/ and record a new decision with rationale. Update docs/STATUS.md and relevant docs in the same PR. Include tests, environment, limitations, and a verified preview URL when available. Do not fabricate successful tests or deployed URLs.
