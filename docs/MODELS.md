# Candidate models and licenses

**Status: candidates, not validated or approved release artifacts.**

| Task | Quality candidate | Lightweight/first-spike candidate |
| --- | --- | --- |
| Classification | [timm MobileNetV4 Conv Medium](https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k) | [timm MobileNetV4 Conv Small](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k) |
| Detection | [D-FINE Small COCO](https://huggingface.co/ustc-community/dfine-small-coco) | [D-FINE Nano COCO](https://huggingface.co/ustc-community/dfine-nano-coco) |

Prefer a **complete pretrained detector** with trained localization/decoder weights over a classifier backbone plus newly initialized detector head. Do not substitute Objects365-derived checkpoints without separately verifying their commercial terms.

Each release candidate requires: immutable source revision, checkpoint filename and SHA-256, exact weight and source-code licenses, training-data provenance, NOTICE/attribution obligations, reproducible conversion, inference parity, backward-gradient/weight-update evidence, fine-tuning quality and memory/runtime measurements.

The model cards display Apache-2.0, but that alone does not finish release-level provenance and license review. Verify exact downloadable artifacts and dependencies before shipping. Never silently auto-update weights.

Upstream benchmark scores are not browser benchmarks or guarantees of downstream accuracy.
