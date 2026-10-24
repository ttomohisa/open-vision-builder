# jax-js ONNX GridSample correction

The developer detector operator tests use `@jax-js/onnx@0.1.2`, installed from
the npm artifact fixed by `package-lock.json`. The package is MIT-licensed by
Eric Zhang (2025). Its original `LICENSE` is retained without modification.

`scripts/patch-jax-onnx-grid-sample.mjs` carries the original MIT notice and
changes four index expressions in the installed ESM and CJS distributions.
The upper neighbor is formed from the **unclipped** floor coordinate, then
clipped independently. This fixes left/top zero-padding forward and VJP results.
There is no other operator, dependency-version or model-recipe replacement.

| File | Original SHA-256 | Patched SHA-256 |
| --- | --- | --- |
| `dist/index.js` | `7ef273af056e1021d668b873c5200027f6a3b326cef50972189e7446a393a7a1` | `362da04f6925eebebf6c2c02d938e3d4f23a610e2c262427b98c50da59b24f51` |
| `dist/index.cjs` | `166276ea065e41b88ed6cee935451903f4e0c4db42f8c0c7709aa8ef60d07213` | `22cd8f1b00ad125d40dbf4cde7af35c169eccf48b3abcc3529966047ddafa893` |
| `LICENSE` | `b9de91fdc4f72bb4e49ad8b57306163bea46f6da7293c08f3b28a922727a952c` | unchanged |

Run `npm ci --ignore-scripts`, then `npm run prepare:deps`. This explicit step
validates both bundles, the package version, and LICENSE before writing.
Unknown bytes and half-applied patches fail closed. `npm run check:deps` checks
without writing and is required by build, dev, tests and the detector probe.
Reinstall the exact lockfile before repairing a half-applied patch.

The original failure is recorded in
[the P1a probe](../../docs/verification/2026-10-24-dfine-probe.md).
Tests compare real `ONNXModel` execution and both input gradients with independent
PyTorch fixtures. The fix is temporary local maintenance, not a claim that it
has been accepted or published upstream. A package upgrade must re-audit the
license, both module formats and numerical regressions before removing it.

Primary artifact: <https://unpkg.com/@jax-js/onnx@0.1.2/dist/index.js>.
