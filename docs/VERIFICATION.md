# Technical verification gates

**All gates: NOT RUN.** Never mark a model supported based solely on inference success.

| Gate | Required evidence |
| --- | --- |
| G0 | Pin exact source, weight hashes, dependencies, license and provenance |
| G1 | Reference PyTorch vs browser forward outputs and preprocessing agree within tolerances defined before running |
| G2 | Gradients reach intended pretrained weights; optimizer changes them and frozen weights stay fixed |
| G3 | Detection updates **pretrained localization** weights; matching/losses work and boxes improve on small training fixtures |
| G4 | Held-out custom-class data evaluation; compare against equivalent Python training recipe |
| G5 | Class-count changes, empty images, multiple objects, small boxes, aspect ratios and invalid labels |
| G6 | Browser-generated complete ONNX passes independent ONNX Runtime Web and Python CPU inference parity |
| G7 | No hidden dependence on training tensors, browser cache, remote model download or operator conversion API |
| G8 | Measure actual browser/GPU, time, peak memory, cancellation, checkpoint and resume |
| G9 | Verify no user images, labels, gradients or weights are transmitted to the operator |

Record fixture licenses, input hashes, environment, commands, numerical tolerances, outcomes and failures. No invented benchmark or unverified success. Inability to fine-tune a candidate requires documenting the blocker and evaluating alternatives, not silently dropping ONNX or true detection training.
