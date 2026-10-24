# D-FINE の前提演算: GridSample の境界プローブ

実施日: 2026-10-24 UTC。

## 判定と範囲

**演算単体の判定は FAIL。D-FINE のブラウザー学習対応は保留する。**

`@jax-js/onnx@0.1.2` の `GridSample` は、左端・上端・左上の補間で手計算した値と一致しなかった。
特徴値に対する勾配とサンプリング座標に対する勾配も不一致だった。
内側・右端・下端の今回の点では、値と両方の勾配が一致した。

これは検出器を構成する演算の前提確認である。**D-FINE Nano/Small の完全なモデルを読み込んだり、
位置推定の重みを更新したりした結果ではない。** [VERIFICATION](../VERIFICATION.md) の検出モデル用
G0–G9 はこのプローブによって合格にならず、引き続き NOT RUN とする。

## 再現方法と固定条件

依存関係を `package-lock.json` から導入した環境で、リポジトリーのルートから実行する。

```sh
npm ci
npm run probe:detector
```

[実行スクリプト](../../scripts/probe-dfine.mjs) は完全な小型 ONNX をメモリー内で生成し、
公開パッケージのローダーと `grad` を直接呼ぶ。画像・モデルのダウンロードやユーザーデータは使わない。
標準出力の JSON に環境、依存バージョン、期待値、実際の値、最大絶対誤差を記録する。

**記録した依存バージョンでは終了コード 1 が期待される。** 不一致を成功へ変換するオプションは設けず、
通常のアプリ単体テストとは別のコマンドにした。将来の修正版で全項目が許容差内に収まる場合は 0 を返す。

| 項目 | 実行条件 |
| --- | --- |
| Node.js | 24.19.0 |
| OS / architecture | Linux / x64 |
| jax-js backend | WASM |
| `@jax-js/jax` | 0.1.27 |
| `@jax-js/onnx` | 0.1.2 |
| `@bufbuild/protobuf` | 2.16.0 |
| `onnx-buf` | 1.20.0-1 |
| ONNX IR / opset | 9 / 17 |
| 演算 | `GridSample`, `mode=bilinear`, `padding_mode=zeros`, `align_corners=0` |
| 入力 | FP32、特徴 `[1,1,4,4]`、座標 `[1,1,1,2]` |
| 出力 | FP32 `[1,1,1,1]` |
| 生成 ONNX SHA-256 | `461af9a276487aa0440a0ef3461ed4d3347f01f334bb9e78db7a58038d082404` |

実行前に、出力値・特徴勾配・座標勾配それぞれの最大絶対誤差を **`1e-5` 以下**と定義した。
この閾値は小さな手計算 fixture 用であり、完全なモデルの許容差に流用しない。
選択した座標は微分の折点となる整数 pixel 座標を避けている。

## 期待値の導出

特徴テンソルは行優先で `[10,20,…,160]` を `[1,1,4,4]` に並べる。
`align_corners=0` なので、正規化座標 `(gx,gy)` の pixel 座標は
`px = 2*gx + 1.5`, `py = 2*gy + 1.5` となる。
範囲外の近傍値を 0 とした通常の四近傍 bilinear 補間から、出力と勾配を独立に計算した。
スクリプト内の期待値はこの固定値であり、被検証のライブラリーから生成していない。
[ONNX GridSample の定義](https://onnx.ai/onnx/operators/onnx__GridSample.html#gridsample-16)

例として左端の `(gx,gy)=(-7/8,-3/8)` は `(px,py)=(-1/4,3/4)` に当たる。
左側の二近傍は 0、右側の二近傍は第 0 列の `10` と `50` なので、
期待する出力は `0.75 * (0.25*10 + 0.75*50) = 30` となる。
特徴勾配は行 0・列 0 に `3/16`、行 1・列 0 に `9/16`、他の位置は 0。
正規化座標に対する勾配は `[80,60]` となる。

## 実行結果

| 点 | 正規化座標 | 期待出力 | 実際の出力 | 特徴勾配の最大絶対誤差 | 座標勾配の期待値 | 座標勾配の実際値 | 判定 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 内側 | `[-3/8,-3/8]` | 47.5 | 47.5 | 0 | `[20,80]` | `[20,80]` | PASS |
| 左端 | `[-7/8,-3/8]` | 30 | 37.5 | 0.5625 | `[80,60]` | `[100,60]` | FAIL |
| 上端 | `[-3/8,-7/8]` | 13.125 | 43.125 | 0.5625 | `[15,35]` | `[15,115]` | FAIL |
| 右端 | `[7/8,-3/8]` | 52.5 | 52.5 | 0 | `[-140,60]` | `[-140,60]` | PASS |
| 下端 | `[-3/8,7/8]` | 103.125 | 103.125 | 0 | `[15,-275]` | `[15,-275]` | PASS |
| 左上 | `[-7/8,-7/8]` | 5.625 | 33.75 | 0.5625 | `[15,15]` | `[90,90]` | FAIL |

左端では実装が第 0 列の代わりに第 1 列を参照し、特徴勾配も第 1 列へ流れた。
配布された実装では `floor(px)` を先に画像内へ clip してから、その値に 1 を加えて隣の座標を作っている。
`floor(px)=-1` の場合、必要な隣接座標は 0 だが、この順序では 1 になる。
上端でも同じ処理がある。この実装上の経路は今回観測した不一致と整合する。
[0.1.2 の配布コード](https://unpkg.com/@jax-js/onnx@0.1.2/dist/index.js)

## Nano/Small 候補への関係

| 公開候補 | 調査した固定 revision | 設定 |
| --- | --- | --- |
| `ustc-community/dfine-nano-coco` | `066438d3d8f0da137a37b38fdf3368fd4afceced` | `decoder_method="default"`, `decoder_n_points=[6,6]` |
| `ustc-community/dfine-small-coco` | `f79e65b5fbb33ceb9d3ebba042955d7410c608f8` | `decoder_method="default"`, `decoder_n_points=[3,6,3]` |

設定の一次資料:

- [Nano config](https://huggingface.co/ustc-community/dfine-nano-coco/blob/066438d3d8f0da137a37b38fdf3368fd4afceced/config.json)。取得バイト列の SHA-256 は `7edf2d96d3deb10aa4ec920be0e8e444b617454545bb58093e32bd8e2b96fa2f`。
- [Small config](https://huggingface.co/ustc-community/dfine-small-coco/blob/f79e65b5fbb33ceb9d3ebba042955d7410c608f8/config.json)。取得バイト列の SHA-256 は `9f206c4c935c67c957af2f15c3333b4ff224e72b1881bc7bbde9f873edf4448e`。

両候補は `discrete` ではなく `default` を指定している。
[Transformers v4.56.0 の D-FINE 実装](https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/models/d_fine/modeling_d_fine.py)
では、その設定を multi-scale deformable attention に渡し、`default` 分岐で
`grid_sample` の bilinear / zeros / `align_corners=False` を使用する。

ネイティブの D-FINE でも、調査 commit
`956d1709314c2c6a4df6f34de232054578a7449f` の
[Nano](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/configs/dfine/dfine_hgnetv2_n_coco.yml) と
[Small](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/configs/dfine/dfine_hgnetv2_s_coco.yml) は
[共通設定](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/configs/dfine/include/dfine_hgnetv2.yml)
の `cross_attn_method: default` を引き継ぐ。
[attention の実装](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/utils.py)
も同じ bilinear / zeros / `align_corners=False` の経路を持つ。

今回の正規化座標はすべて `[-1,1]` 内であり、D-FINE の `sampling_locations` へ戻すと
`(grid+1)/2` で `[0,1]` 内に入る。境界の補間は候補の演算契約に含まれる。
ただし、公開チェックポイントと実画像で該当座標がどれほど出現するか、最終 boxes・scores・AP に
どれほど影響するかは、このプローブでは測定していない。

## 次に確認すること

- 演算を修正した版または対応する別実装で、この固定 fixture と元実装の forward/backward を再比較する。
- 採用する完全な検出グラフについて operator、domain、opset、dtype、外部データの inventory を作る。
- 学習対象の pretrained localization 重みを明示的な入力へ移し、その勾配と実更新を検証する。
- deploy グラフと training forward の補助出力・matching・detach・損失の差を実装レシピで扱う。
- その後に D-FINE 用 G0–G9 を実行する。分類ヘッド学習の成功を検出器学習の合格へ流用しない。

この記録にはブラウザー WebGPU の結果、D-FINE 全体の forward、検出器の追加学習、
モデル精度、GPU メモリー、学習時間の測定結果を含めていない。
