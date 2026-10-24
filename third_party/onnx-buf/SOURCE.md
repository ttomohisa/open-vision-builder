# onnx-buf and ONNX schema notices

- Package: `onnx-buf@1.20.0-1`; npm metadata declares MIT.
- Package-source tag: `onnx-buf/v1.20.0-1` in the author's `ekzhang/protobuf-es-pkg` repository.
- Package-source revision: `b75855e925d63d1f4d2ab3ccf32432704ae5debb`.
- Generation input: ONNX `v1.20.0`, revision `bca0315ff3e56bb5847509ca32fadb8d87f680e1`.

The npm package's only JavaScript file, `dist/onnx_pb.js`, begins with
`SPDX-License-Identifier: Apache-2.0`. It is generated from ONNX's `onnx.proto`.
The package-source tag's generation configuration is preserved in `buf.gen.yaml`:
<https://raw.githubusercontent.com/ekzhang/protobuf-es-pkg/b75855e925d63d1f4d2ab3ccf32432704ae5debb/buf.gen.yaml>.
Its SHA-256 is `3012bc95984c2e182350479bf903cd23020913033bf5542fb37e72c6b89e6598`.

The package's MIT metadata does not replace the generated schema's Apache
license. Neither its npm tarball nor the package-source tag supplies a separate
MIT LICENSE file. We retain the metadata and the actual generated code's Apache
notice, and do not substitute another project's MIT license for the package.

`LICENSE-ONNX` is copied unchanged from the exact schema release:
<https://raw.githubusercontent.com/onnx/onnx/bca0315ff3e56bb5847509ca32fadb8d87f680e1/LICENSE>.
Its SHA-256 is `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
ONNX attributes this source to Copyright (c) ONNX Project Contributors in
<https://github.com/onnx/onnx/blob/bca0315ff3e56bb5847509ca32fadb8d87f680e1/REUSE.toml>.

Retrieved and checked on 2026-10-24. The npm tarball URL and integrity are in
`package-lock.json`; the npm metadata's repository URL still points to jax-js,
so the explicit package-source and generation-input revisions above are retained.
