# Additional notices for code origins in jax-js

`@jax-js/jax@0.1.27` and `@jax-js/optax@0.1.2` ship their MIT LICENSE files in
their npm tarballs. The collector includes those original files.

For jax `0.1.27`, the official `jax/v0.1.27` tag resolves to
`d162baa7cfdb9f94f6a77b0eb6c60853db641f4f`. The shipped package includes code whose
upstream comments identify these additional origins:

- [Shape tracking](https://github.com/ekzhang/jax-js/blob/d162baa7cfdb9f94f6a77b0eb6c60853db641f4f/src/shape.ts) is based on tinygrad.
- [Convolution pooling](https://github.com/ekzhang/jax-js/blob/d162baa7cfdb9f94f6a77b0eb6c60853db641f4f/src/frontend/convolution.ts) references tinygrad `v0.10.3`, revision `9b14e8c3cd26c3687870e87971595f57ab512566`.
- [Tuning](https://github.com/ekzhang/jax-js/blob/d162baa7cfdb9f94f6a77b0eb6c60853db641f4f/src/tuner.ts) references tinygrad revision `685d5c46df`; its MIT LICENSE is byte-identical to the copy from `v0.10.3`.
- [Synchronous WebGPU reading](https://github.com/ekzhang/jax-js/blob/d162baa7cfdb9f94f6a77b0eb6c60853db641f4f/src/backend/webgpu/reader.ts) credits a TensorFlow.js technique and links to `tfjs-v4.22.0`, revision `e5d5e9371ed1fd0a4df6d7cd0b947d2a820cefd7`.

`ThirdPartyNotices.txt` retains tinygrad's MIT notice and the referenced
TensorFlow.js source's copyright plus Apache-2.0 license. These are additional
attributions for source origins; they do not mean the full tinygrad or TensorFlow.js
packages are bundled or selected as this application's training engine.
The original license URLs and SHA-256 hashes are included alongside the text.
The combined notice SHA-256 is `0807009885a0ad467e7ffefc20af55dc693c14f781dc1001d92b5681873940f2`.

Retrieved and checked on 2026-10-24. This supplements the npm package notices
without changing their license or the application's license.
