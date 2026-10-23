# Product requirements

**Concept:** An open, browser-based alternative for people who liked Custom Vision.

## Confirmed requirements
1. Static Cloudflare hosting; no operator API, DB, accounts or training infrastructure.
2. Local browser annotation, training and evaluation using the user's hardware, without uploading their data.
3. Image classification and bounding-box object detection, including custom classes.
4. Transfer learning/fine-tuning from commercially usable pretrained weights. Detection must retain pretrained localization capability.
5. **Complete ONNX export in the initial release**, produced in the browser after training, directly usable by Python/ONNX Runtime without conversion.
6. Package labels, preprocessing/postprocessing specification, inference examples, license notices and self-hosting/deployment starter kits. Users deploy the model themselves.
7. Local project save/restore and backups, with interruption/restart handling.
8. Development through GitHub PRs reviewed and merged by the repository owner, with GitHub Pages previews for working application PRs.
9. Separate product and repository from Browser Kitty.

## Proposed initial scope
Single-label classification; multi-box detection; image import, label/box editor, train/validation/test splits, training and evaluation, mistake review, re-training, project backup and model export.

No guarantee of every browser, smartphone, NPU or edge accelerator. Begin by evaluating desktop Chrome/Edge with WebGPU. No server-side fallback that silently violates the product constraints.
