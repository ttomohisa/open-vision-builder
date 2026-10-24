# GridSample の境界修正と独立 forward / backward 検証

実施日: 2026-10-24 UTC。

## 判定と範囲

**修正した GridSample の演算単体ゲートは PASS。** Node.js の ESM / CommonJS で手計算6ケースと
PyTorch 参照12ケースを、実際のブラウザー WASM で同じ PyTorch 参照12ケースを確認した。
出力値、特徴テンソルに対する勾配、サンプリング座標に対する勾配が、実行前に固定した許容差内で一致した。
[元の配布実装の FAIL 記録](2026-10-24-dfine-probe.md) は履歴として保持する。

これは `GridSample` の前提確認であり、**検出モデル全体の G1 / G2 を合格にする結果ではない。**
凍結した prefix の特徴を元実装と比較する別の G1 検証には FAIL が残る。
事前学習済み最終ヘッドの更新、完全 ONNX のブラウザー書き出し、Python の独立推論はそれぞれ別の証拠で判断する。
P1b の全体判定は [STATUS](../STATUS.md) と [VERIFICATION](../VERIFICATION.md) を参照し、
この演算の PASS を D-FINE 全体の追加学習成立や製品の完成として扱わない。

## 修正内容と配布物の固定

上流 `@jax-js/onnx@0.1.2` は、`floor(px)` / `floor(py)` を画像内へ clip してから 1 を加え、
隣接 pixel の座標を作っていた。raw floor が `-1` のとき、必要な隣接座標は `0` だが、
この順序では `1` を参照する。左端・上端での出力と勾配の誤りを、修正前の独立 fixture で再現した。

[修正スクリプト](../../scripts/patch-jax-onnx-grid-sample.mjs) は、ESM / CommonJS の各配布物の
4行だけを変更する。`x1` / `y1` は **clip 前の floor に 1 を加えた座標**から独立に clip し、
整数へ変換する。`x0` / `y0` を作る際の `.ref` で raw floor の参照を残し、jax-js の tensor 所有権を保つ。
ONNX の公開ローダー `ONNXModel` 内で使われる実装を直しており、未使用の別演算やグローバルな
実行時差し替えを追加していない。依存バージョン・他の演算・モデルレシピは変更していない。

公開 API に演算差し替えの拡張点がないため、ローダーを複製する代わりに、この固定配布物へ明示的な
局所修正を適用する。上流へ採用・公開済みの修正という意味ではなく、依存更新時には再監査する。

| 対象 | 修正前 SHA-256 | 修正後 SHA-256 |
| --- | --- | --- |
| `@jax-js/onnx@0.1.2/dist/index.js` | `7ef273af056e1021d668b873c5200027f6a3b326cef50972189e7446a393a7a1` | `362da04f6925eebebf6c2c02d938e3d4f23a610e2c262427b98c50da59b24f51` |
| `@jax-js/onnx@0.1.2/dist/index.cjs` | `166276ea065e41b88ed6cee935451903f4e0c4db42f8c0c7709aa8ef60d07213` | `22cd8f1b00ad125d40dbf4cde7af35c169eccf48b3abcc3529966047ddafa893` |
| 同パッケージの `LICENSE` | `b9de91fdc4f72bb4e49ad8b57306163bea46f6da7293c08f3b28a922727a952c` | 変更なし |

一次配布物: [ESM](https://unpkg.com/@jax-js/onnx@0.1.2/dist/index.js)・
[CommonJS](https://unpkg.com/@jax-js/onnx@0.1.2/dist/index.cjs)。
上流の MIT License、Copyright (c) 2025 Eric Zhang を依存パッケージ内で保持し、
修正スクリプトにも元の MIT 表示を含めた。[出典と保守方針](../../third_party/jax-js-onnx/SOURCE.md) を参照。

修正前にパッケージ名・バージョン・LICENSE・**両方のファイル全体の SHA-256**を確認し、
生成予定の修正後ハッシュも確認してから書き込む。適用済みなら再適用しても内容を変えない。
未知の内容、別バージョン、変更されたライセンス、ESM / CommonJS の片方だけが修正済みの状態は停止する。
`--check` は書き込みを行わず、両方が承認済みの修正後バイト列であることを要求する。
途中状態を自動修復せず、lockfile から再導入して明示的に準備し直す。

## 固定した実行条件と参照値

| 項目 | 条件 |
| --- | --- |
| Node.js / OS | `24.19.0` / Linux x64 |
| ブラウザー | Headless Chromium `156.0.8078.4` / Linux x86_64 |
| 実行 backend | Node / browser とも WASM |
| ブラウザー isolation | `crossOriginIsolated=false` |
| `@jax-js/jax` | `0.1.27` |
| `@jax-js/onnx` | `0.1.2` + `grid-sample-unclipped-neighbors-v1` |
| `@bufbuild/protobuf` / `onnx-buf` | `2.16.0` / `1.20.0-1` |
| PyTorch 参照 | `2.6.0+cpu`、CPU、FP32、deterministic algorithms、1 thread |
| ONNX IR / opset | `9` / `17` |
| 演算契約 | 4D feature / 4D grid、`mode=bilinear`、`padding_mode=zeros`、FP32 |
| `align_corners` | 手計算 fixture は `0`、PyTorch fixture は `0` と `1` |

### 手計算6ケース

[元のプローブ](2026-10-24-dfine-probe.md#期待値の導出) と同じ、内側・左端・上端・右端・下端・左上を使う。
特徴 `[1,1,4,4]`、grid `[1,1,1,2]`、目的関数 `sum(output)` の出力と両入力勾配を比較する。
この6ケースは整数 pixel の微分の折点を避けている。生成した小型 ONNX の SHA-256 は
`461af9a276487aa0440a0ef3461ed4d3347f01f334bb9e78db7a58038d082404` で、元の記録と同じ。

### PyTorch 参照12ケース

[独立した生成スクリプト](../../scripts/generate_grid_sample_reference.py) が
`torch.nn.functional.grid_sample` と autograd で期待値を作り、
[小型 JSON fixture](../../tests/fixtures/grid-sample-pytorch.json) として保存する。
期待値生成に jax-js のコードや出力は使わない。
目的関数は `sum(output * cotangent)`。符号と値が場所ごとに異なる cotangent を使い、
複数 batch / channel の添字と、同じ入力位置へ集まる勾配も確認する。

| ケース群 | Feature `[N,C,H,W]` | Grid `[N,Hout,Wout,2]` | 各群の条件 |
| --- | --- | --- | --- |
| `edges-corners-outside` | `[1,2,4,4]` | `[1,4,4,2]` | 4辺、4隅、範囲外、内側、正規化座標の正確な四隅 |
| `nonsquare-multiple-batches-channels` | `[2,3,3,5]` | `[2,2,5,2]` | 非正方形、異なる batch の grid、複数 channel / 出力行列 |
| `height-one` | `[2,2,1,5]` | `[2,3,3,2]` | 高さ1、内側・境界・範囲外 |
| `width-one` | `[2,2,5,1]` | `[2,3,3,2]` | 幅1、内側・境界・範囲外 |
| `one-pixel` | `[2,2,1,1]` | `[2,3,3,2]` | 高さ・幅とも1 |
| `repeated-samples` | `[2,2,4,3]` | `[2,3,2,2]` | 同じ位置への反復 sampling と勾配の加算 |

各群を `align_corners=0` / `1` で実行し、計12ケースとする。
`align_corners=1` の正確な四隅は微分の折点に当たるため、ここでは PyTorch autograd が返す勾配を
直接参照する。折点における中央差分との一致を主張する試験ではない。

| 参照物 | SHA-256 |
| --- | --- |
| `scripts/generate_grid_sample_reference.py` | `b4ae97be9ee2520d814fd5d440e3b03753ca54616ff3353efdd122e833fb97a6` |
| `tests/fixtures/grid-sample-pytorch.json` | `34258d7f9d0e3b5dd8da6a5c55bbf6a482cecfa1b53bfb0e28369a6a103c04b2` |
| 検証用 browser bundle | `80898be1445f0aaa66703924565fd9e4c7f9286090ba541503154b2747294b13` |

### 実行前に固定した許容差

手計算6ケースは、出力・特徴勾配・座標勾配それぞれで最大絶対誤差 `1e-5` 以下とする。
PyTorch 参照12ケースは、以下を **全要素**で満たすことを要求する。

`abs(actual - expected) <= atol + rtol * abs(expected)`

| 比較対象 | `atol` | `rtol` |
| --- | ---: | ---: |
| 出力値 | `1e-5` | `1e-5` |
| 特徴テンソルに対する勾配 | `2e-5` | `1e-5` |
| grid 座標に対する勾配 | `2e-5` | `1e-5` |

shape の不一致と非有限値は不合格とする。拡張 fixture の実行前にこの許容差を決め、修正前の失敗を
確認した後も緩和していない。完全な検出モデルの比較許容差には流用しない。

## 実行結果

| 実装と実行環境 | 手計算6ケース | PyTorch 参照12ケース | 判定 |
| --- | --- | --- | --- |
| 上流未修正 / Node ESM | 3 PASS、3 FAIL | 5 PASS、7 FAIL | FAIL |
| 局所修正後 / Node ESM | 6 / 6 PASS | 12 / 12 PASS | PASS |
| 局所修正後 / Node CommonJS | 6 / 6 PASS | 12 / 12 PASS | PASS |
| 局所修正後 / browser ESM / WASM | この browser 試験では対象外 | 12 / 12 PASS | PASS |

修正前には Node ESM / CommonJS の数値テストを両方実行し、どちらも数値比較の assertion が FAIL になった。
上流未修正の詳細 JSON は ESM で保存した。修正後、手計算6ケースの出力・特徴勾配・座標勾配は
Node ESM / CommonJS とも最大絶対誤差 **0**。

PyTorch 参照12ケースの最大絶対誤差は次のとおり。3実行環境で同じ値だった。

| 実行環境 | 出力値 | 特徴勾配 | 座標勾配 |
| --- | ---: | ---: | ---: |
| Node ESM / WASM | `1.1920928955078125e-6` | `2.384185791015625e-7` | `1.9073486328125e-6` |
| Node CommonJS / WASM | `1.1920928955078125e-6` | `2.384185791015625e-7` | `1.9073486328125e-6` |
| Chromium / WASM | `1.1920928955078125e-6` | `2.384185791015625e-7` | `1.9073486328125e-6` |

全要素における `abs(error) / (atol + rtol * abs(expected))` の最大値は
`0.042151333234024785` で、合格上限の `1` 以下だった。

[Node の回帰テスト](../../tests/grid-sample.test.ts) と
[修正適用のテスト](../../tests/patch-jax-onnx.test.ts) は計8テストが PASS。
後者は未準備の `--check`、両形式への適用・冪等性、CJS が未知の内容の場合に ESM も書き込まないこと、片側だけの修正、
別バージョン、変更された LICENSE の拒否を確認する。PyTorch fixture の再生成 `--check` も PASS。

[ブラウザー回帰テスト](../../tests/grid-sample.browser.spec.ts) は1テストで12ケースを実行し PASS。
[test 専用の entry](../../tests/fixtures/grid-sample-browser.ts) を Vite で bundle 化し、
実際にインストールされた `ONNXModel` を実行した。製品 UI に診断用の操作を加えていない。
ローカル subpath `/open-vision-builder/pr-local/` で、検証 HTML と bundle の同一 origin への GET 2件だけを観測した。
予期しない通信と page error はともに0件。これは当該試験の通信記録であり、製品全体の G9 判定とは分ける。

## 再現と明示的な準備

リポジトリーの root で実行する。Python 参照の再確認には、開発用環境の固定した
`torch==2.6.0+cpu` を使う。利用者のブラウザー学習に Python は要求しない。

```sh
npm ci --ignore-scripts
npm run prepare:deps
npm run check:deps
npm run probe:detector
node scripts/probe-dfine.mjs --format cjs
npm test -- tests/grid-sample.test.ts tests/patch-jax-onnx.test.ts
.venv/bin/python scripts/generate_grid_sample_reference.py --check
```

`prepare:deps` は `node scripts/patch-jax-onnx-grid-sample.mjs`、
`check:deps` は同スクリプトの `--check` に対応する。install lifecycle hook には依存しない。
`npm run dev`、`npm run build`、`npm test`、`npm run test:browser`、`npm run probe:detector` は
準備済みの依存を検査し、未修正・未知の内容での実行を停止する。

ブラウザー試験は Playwright Chromium を導入し、同じ base で build してから実行する。
この単体演算試験は公開検出モデルのダウンロードやユーザー画像を必要としない。

```sh
OVB_BASE=/open-vision-builder/pr-local/ npm run build
OVB_BASE=/open-vision-builder/pr-local/ npm run test:browser -- tests/grid-sample.browser.spec.ts
```

実測時の browser runner は `npx playwright test tests/grid-sample.browser.spec.ts` で、
同じ `OVB_BASE` と環境に導入済みの `PLAYWRIGHT_BROWSERS_PATH` を指定した。

上流未修正の負例を再現する場合は、新しい導入直後、**修正適用前**に直接 probe を呼ぶ。

```sh
npm ci --ignore-scripts
node scripts/probe-dfine.mjs
```

この直接実行は既知の数値不一致を示し、終了コード1になる。通常の `npm run probe:detector` は先に
`check:deps` が走るため、未準備状態での数値の負例確認には使わない。
確認後に `npm run prepare:deps` を実行すれば、修正後の通常 probe は終了コード0になる。

## 証拠ファイルと未検証範囲

生の測定結果は ignored の `artifacts/p1b/` に出力し、数値の要約・参照 fixture・実行コードを本 PR に残す。

| ファイル | 内容 |
| --- | --- |
| `grid-sample-upstream.json` | 元の手計算6ケースの上流 FAIL |
| `grid-sample-upstream-expanded.json` | 手計算6 + PyTorch 12ケースの上流 ESM FAIL |
| `grid-sample-tests-red.log` | 修正前の ESM / CommonJS 数値テスト FAIL |
| `grid-sample-patch.json` | パッケージ・修正 ID・両ファイルとライセンスのハッシュ |
| `grid-sample-patched-esm.json` / `grid-sample-patched-cjs.json` | 修正後の値・勾配・比較結果 |
| `grid-sample-tests-green.log` | 修正後の8テスト PASS |
| `grid-sample-browser.json` | browser / backend・比較結果・bundle hash・通信記録 |

WebGPU、5D GridSample、他の補間・padding mode、全座標・全 shape、完全な D-FINE training forward、
元モデルとの凍結特徴一致、精度、速度、ピークメモリーはこの演算試験の認定範囲に含めない。
今後の依存更新では元の FAIL fixture と独立参照を残し、両 module format と browser の値・両勾配を再確認する。
