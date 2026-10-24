"""Independent, deterministic CPU PyTorch references for the ONNX GridSample gate.

Developer/CI use only. No images, model weights, network access, or user data.
The JSON is deliberately small enough to review and keep with the numerical tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests/fixtures/grid-sample-pytorch.json"
TORCH_VERSION = "2.6.0+cpu"


def patterned_values(shape: list[int], multiplier: int, offset: int, modulus: int) -> torch.Tensor:
    # Dyadic fractions keep input/cotangent serialization exact in FP32.
    values = [((index * multiplier + offset) % modulus - modulus // 2) / 8
              for index in range(math.prod(shape))]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def pixel_grid(points: list[tuple[float, float]], height: int, width: int,
               align_corners: bool) -> list[tuple[float, float]]:
    def normalize(pixel: float, size: int) -> float:
        return (2 * pixel / (size - 1) - 1) if align_corners else ((2 * pixel + 1) / size - 1)
    return [(normalize(x, width), normalize(y, height)) for x, y in points]


def reference_case(name: str, feature_shape: list[int], grid_shape: list[int],
                   points: list[tuple[float, float]], align_corners: bool) -> dict:
    features = patterned_values(feature_shape, 17, 3, 53).requires_grad_()
    grid = torch.tensor(points, dtype=torch.float32).reshape(grid_shape).requires_grad_()
    output = functional.grid_sample(features, grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=align_corners)
    cotangent = patterned_values(list(output.shape), 11, 5, 19)
    # Nonuniform signed cotangents check channel sums and repeated-gather accumulation.
    (output * cotangent).sum().backward()
    assert features.grad is not None and grid.grad is not None

    def flat(tensor: torch.Tensor) -> list[float]:
        return tensor.detach().flatten().tolist()

    return {
        "name": f"{name}-align{int(align_corners)}",
        "alignCorners": int(align_corners),
        "featureShape": feature_shape,
        "gridShape": grid_shape,
        "outputShape": list(output.shape),
        "features": flat(features),
        "grid": flat(grid),
        "cotangent": flat(cotangent),
        "expected": {
            "output": flat(output),
            "featureGradient": flat(features.grad),
            "gridGradient": flat(grid.grad),
        },
    }


def generate_reference() -> dict:
    if torch.__version__ != TORCH_VERSION:
        raise RuntimeError(f"Expected pinned torch {TORCH_VERSION}; found {torch.__version__}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = []
    for align_corners in (False, True):
        # Includes four partially padded corners, four edges, fully outside,
        # interior points and exact normalized corners. Pixel knots are avoided
        # except exact align_corners=True corners, whose PyTorch derivative is
        # recorded explicitly (these points are not a finite-difference check).
        boundary = pixel_grid([
            (-0.25, 0.75), (0.75, -0.25), (3.25, 0.75), (0.75, 3.25),
            (-0.25, -0.25), (3.25, -0.25), (-0.25, 3.25), (3.25, 3.25),
            (-1.25, 0.75), (0.75, 4.25), (0.75, 0.75), (1.25, 2.75),
        ], 4, 4, align_corners) + [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        cases.append(reference_case("edges-corners-outside", [1, 2, 4, 4], [1, 4, 4, 2],
                                    boundary, align_corners))

        nonsquare = pixel_grid([
            (-0.25, 0.75), (0.75, -0.25), (4.25, 0.75), (0.75, 2.25),
            (-0.25, -0.25), (4.25, 2.25), (-1.25, 0.75), (0.75, 3.25),
            (0.75, 0.75), (3.25, 1.25),
        ], 3, 5, align_corners)
        cases.append(reference_case("nonsquare-multiple-batches-channels", [2, 3, 3, 5],
                                    [2, 2, 5, 2], nonsquare + nonsquare[3:] + nonsquare[:3],
                                    align_corners))

        degenerate_points = [(-2.5, -2.5), (-1, -1), (-0.7, 0.2), (0.2, -0.7),
                             (0.65, 0.25), (1, 1), (1.4, -0.3), (-0.3, 1.4), (2.5, 2.5)]
        for name, shape in (("height-one", [2, 2, 1, 5]),
                            ("width-one", [2, 2, 5, 1]),
                            ("one-pixel", [2, 2, 1, 1])):
            cases.append(reference_case(name, shape, [2, 3, 3, 2],
                                        degenerate_points + list(reversed(degenerate_points)),
                                        align_corners))

        repeated = [(-0.95, -0.85), (-0.95, -0.85), (0.2, 0.3), (0.2, 0.3),
                    (1.2, 0.7), (1.2, 0.7)]
        cases.append(reference_case("repeated-samples", [2, 2, 4, 3], [2, 3, 2, 2],
                                    repeated + list(reversed(repeated)), align_corners))
    return {
        "schemaVersion": 1,
        "reference": {
            "implementation": "torch.nn.functional.grid_sample + autograd",
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "mode": "bilinear",
            "paddingMode": "zeros",
            "objective": "sum(output * cotangent)",
            "generatorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if the recorded reference differs; do not write")
    args = parser.parse_args()
    serialized = json.dumps(generate_reference(), indent=2, allow_nan=False) + "\n"
    if args.check:
        if args.output.read_text() != serialized:
            raise SystemExit(f"PyTorch GridSample reference differs: {args.output}")
        print(f"Verified PyTorch GridSample reference: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(f"Wrote PyTorch GridSample reference: {args.output}")


if __name__ == "__main__":
    main()
