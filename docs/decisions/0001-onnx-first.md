# ADR 0001 — Browser training and ONNX-first export

Date: 2026-10-24. Status: product direction accepted; model/engine implementation pending verification.

## Context
The initial MobileNetV2 plus fresh detection head plan raised concerns about model age and accuracy. The user also requires ONNX for Python/API/edge usage **from the first release**.

## Decision
- Keep static-only hosting and fully local browser training.
- Make complete browser-generated ONNX a first-release requirement.
- Candidate classification: MobileNetV4 Conv Small/Medium.
- Candidate detection: pretrained D-FINE Nano/Small COCO, including pretrained localization components.
- Compare jax-js and TensorFlow.js by demonstrated gradients, parameter updates, parity and browser resource use, not framework popularity.
- Model-specific exporters are acceptable; generic arbitrary-ONNX fine-tuning is not a v1 requirement.
- Check commercial terms on exact checkpoints and bundled components.
- Prove training and ONNX output before building a broad UI.

## Consequences
ONNX deploy graphs often omit training-specific operations. The training graph, loss, matcher, class-count-dependent layers, parameter mapping and ONNX writer require explicit implementation and independent validation. Failed candidates may be replaced, but product constraints may not be weakened without a new user-approved decision.
