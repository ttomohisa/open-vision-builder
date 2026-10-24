# MobileNetV4 公開モデル準備の検証

検証日: 2026-10-24 UTC。

**結果: 固定した公開 safetensors の SHA-256 照合、完全モデルの strict load、単一 FP32 ONNX の生成、固定入力に対する PyTorch / Python ONNX Runtime CPU の出力一致を確認した。**

これは分類 P1 の開発者側のモデル準備についての証拠である。ブラウザー学習、ブラウザーが書き出したモデル、実画像での分類品質、検出器の学習、製品全ゲートの完了をこの結果から主張しない。[検証計画](../VERIFICATION.md) の全体判定とは区別する。

## 1. 固定した出典

| 項目 | 値 |
| --- | --- |
| モデル | `timm/mobilenetv4_conv_small.e2400_r224_in1k` |
| HF revision | `331fb803779522b685cf942e15f914fb6741c1eb` |
| 元ファイル | `model.safetensors` |
| 元ファイル bytes | `15223016` |
| 元ファイル SHA-256 | `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4` |
| timm | `1.0.15` |
| timm tag の commit | `e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d` |
| モデル単位の条件 | 固定モデルカードに `license: apache-2.0` |

一次資料:

- [固定した重み](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/resolve/331fb803779522b685cf942e15f914fb6741c1eb/model.safetensors)
- [固定したモデルカード](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/blob/331fb803779522b685cf942e15f914fb6741c1eb/README.md)
- [固定した元 config](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/blob/331fb803779522b685cf942e15f914fb6741c1eb/config.json)
- [timm 実装](https://github.com/huggingface/pytorch-image-models/blob/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/timm/models/mobilenetv3.py)
- [timm LICENSE](https://github.com/huggingface/pytorch-image-models/blob/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/LICENSE)

`scripts/prepare_model.py` はファイル全体の SHA-256 とサイズを照合してから safetensors で読み込む。pickle の読み込みは使わない。モデルカード・元 config・LICENSE も固定 URL と SHA-256 で照合する。モデルカードの Apache-2.0 宣言と実装の LICENSE を別々に保持する。

## 2. 実行環境と再現コマンド

実行環境は Python 3.12.14、Linux 6.18.55 x86_64、glibc 2.39。CPUExecutionProvider、PyTorch/ORT とも計算スレッド1で実行した。

| 依存 | 実行した版 |
| --- | --- |
| PyTorch | `2.6.0+cpu` |
| torchvision | `0.21.0+cpu` |
| timm | `1.0.15` |
| ONNX | `1.17.0` |
| ONNX Runtime | `1.20.1` |
| NumPy | `2.2.3` |
| safetensors | `0.5.3` |
| Pillow | `11.1.0` |
| protobuf | `7.36.2` |

推移的な依存も含む30パッケージの解決結果を `scripts/requirements-models.txt` に固定した。開発者または CI が次を実行する。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r scripts/requirements-models.txt
.venv/bin/python scripts/prepare_model.py
.venv/bin/python -m unittest discover -s tests -p '*_test.py'
```

この作業環境では既定の `/root/.cache/uv` が書き込み不可だったため、uv の実行時に `UV_CACHE_DIR=/tmp/ovb-model-uv-cache` を指定した。モデル変換は開発時の公開モデル準備であり、利用者の学習・書き出し手順には含めない。

## 3. モデルと前処理の契約

元モデルを1000クラスのまま構築し、全 state dict を `strict=True` でロードした後、eval / FP32 / requires_grad=False にした。元分類器をロード前に除去する方法は使っていない。

| 対象 | 実際の shape |
| --- | --- |
| 元 `conv_head.weight` | `[1280, 960, 1, 1]` |
| 元 `classifier.weight` | `[1000, 1280]` |
| 入力 `images` | `[1, 3, 224, 224]` |
| 出力 `features` | `[1, 1280]` |
| 出力 `logits` | `[1, 1000]` |

特徴は `forward_head(forward_features(images), pre_logits=True)`。元 config の `num_features=960` は特徴抽出段のチャンネル数であり、分類器直前の1280次元と区別する。完全グラフには元分類器とその出力も残し、ブラウザーはこの1280次元特徴から新しい分類ヘッドを学習する。

今回の入力 recipe は次のとおり。

1. RGBA を `RGB * (alpha/255) + 255 * (1-alpha/255)` で白へ浮動小数点合成する。
2. 全画像を224×224へ bilinear stretch する。
3. source coordinate は `(destination + 0.5) * sourceSize / targetSize - 0.5`。
4. 二つの source index をそれぞれ画像端へ clamp する。合成・補間の途中に8-bit丸めを挟まない。
5. RGB を255で割り、mean `[0.485, 0.456, 0.406]`、std `[0.229, 0.224, 0.225]` で正規化する。
6. 最終の NCHW tensor を FP32 にする。

**この stretch recipe は元 timm の bicubic resize / center crop と異なる。上流の ImageNet 精度を、このアプリの前処理や利用者データでの精度として引き継がない。** この差は生成する NOTICE と `preparation.json` にも記録する。

## 4. 固定 fixture と合格条件

fixture は40×32の自動生成 RGBA 画像で、第三者の写真や利用者のデータを含まない。

```text
r = (x*7 + y*3) % 256
g = (x*2 + y*11) % 256
b = (x*13 + y*5) % 256
a = 255
```

fixture 名は `rgba-pattern-v1`。前処理後の little-endian FP32 入力の SHA-256 は `10137b9e98ddfa54fae3af344530b4169bcc391961fb95a550234ef6b01377c3`。

実行前に [実装計画](../superpowers/plans/2026-10-24-classification-spike.md) で定めた条件を使用した。各要素について `abs(actual - expected) <= 2e-4 + 1e-4 * abs(expected)` を要求する。expected は元 PyTorch 出力。shape 不一致、NaN、Inf、外部重み、予期しない最終分類器は失敗にする。

## 5. 実測結果

| 出力 | 要素数 | 最大絶対誤差 | 誤差 / 許容値の最大比 | 許容外要素 |
| --- | ---: | ---: | ---: | ---: |
| logits | 1000 | `1.1444091796875e-05` | `0.004783392825452879` | 0 |
| features | 1280 | `5.066394805908203e-06` | `0.022992301309232344` | 0 |

両出力が合格した。`onnx.checker.check_model(..., full_check=True)` も通った。準備処理を3回実行し、いずれも同一の ONNX SHA-256 と比較結果になった。これはこの固定環境での再現結果であり、未確認の別 OS や依存版での byte 一致を保証しない。

| 成果物 | 実測値 |
| --- | --- |
| `backbone.onnx` bytes | `15069036` |
| SHA-256 | `28857aef081dbba9e038c04cd8f2dabce28f654c66ffa47af1444179194f21d7` |
| ONNX IR | 8 |
| ONNX opset | 17、標準 domain |
| nodes / initializers | 89 / 94 |
| 演算数 | Conv 46、Relu 30、Add 10、GlobalAveragePool 1、Flatten 1、Gemm 1 |
| 外部重み | なし |
| 浮動小数点重み | FP32 |

元の最終ノードは `/classifier/Gemm`。入力は `features`、`base.classifier.weight`、`base.classifier.bias`。`alpha=1.0`、`beta=1.0`、`transB=1`、`transA` は既定の0。これらを生成後に実物から解析し、manifest と `preparation.json` に記録した。eval BatchNorm は export 時に Conv へ融合され、生成グラフに BatchNormalization ノードはない。

## 6. 成果物と回帰テスト

準備スクリプトは `public/models/mobilenetv4-small/` に次を生成する。

- `backbone.onnx`: 全凍結ネットと元分類器を含む単一 ONNX。
- `model.json`: `src/ml/types.ts` の ModelManifest 契約。
- `reference.json`: NativeReference 契約、元 logits / features、事前許容差、実行版。
- `preparation.json`: 取得元、変換、演算、入力 hash、比較結果、環境の証拠。
- `LICENSE.txt`, `NOTICE.txt`, `MODEL_CARD.md`, `SOURCE_CONFIG.json`: 条件・来歴・変更の記録。

モデルのバイナリは gitignore 対象。元 checkpoint は `.venv/model-source-cache/` にのみ保持し、静的配信用の directory へは入れない。実際に `git check-ignore` で両方を確認した。

`tests/model_scripts_test.py` は11テストが合格した。手計算した半画素補間の端・中央、半透明画素の白合成、生成 fixture、入力 shape、壊れた checkpoint cache の拒否、独立した小さい ONNX の実推論、誤った logits / NaN / label 数の拒否を検証する。fixture と ONNX はテスト内で生成し、ネットワークは使わない。変更時点の `npm test` も36テストが合格した。

`scripts/predict.py` はブラウザーから持ち出した完全 ONNX と report だけを使う開発用の独立検証器。source checkpoint や prepare スクリプトを参照せず、成功・比較結果を JSON で標準出力へ返す。比較失敗や入力不正は終了コード1。

```bash
.venv/bin/python scripts/predict.py --model artifacts/browser/model.onnx --verify-report artifacts/browser/report.json
```

report は `labels: string[]` と `parityFixture: {input: number[], shape: [1,3,224,224], expectedLogits: number[]}` を持つ。上の browser artifact の実検証はブラウザー側の証拠に記録する。本テストでは生成した小さい既知モデルに対してこの CLI の合格と拒否動作を実行した。

## 7. 利用者向けの通常推論例

`public/kit/predict.py` と `README.txt` は配布キットに含める小さなファイル。NumPy、ONNX Runtime、Pillow のみで、通常の PNG/JPEG、完全 ONNX、labels JSON を読み、logits・softmax・クラス名を JSON に出力する。

```bash
pip install onnxruntime numpy Pillow
python predict.py --model model.onnx --image image.png --labels labels.json
```

Pillow では画像 decode と EXIF orientation まで行い、白合成とリサイズを NumPy の浮動小数点で行う。Pillow の8-bit alpha 合成は使わない。通常推論にはモデル準備用の PyTorch/timm/ONNX package を要求しない。

独立した `.venv/kit-runtime` に NumPy 2.2.3、ORT 1.20.1、Pillow 11.1.0 とその推移的依存だけを入れ、`torch`, `timm`, `onnx` が未導入であることを確認した。その環境で、生成 fixture PNG、1000個の検証用クラス名、実際の `backbone.onnx` を CLI に渡して推論した。

結果は1000 logits、softmax の和1.0、元 native logits との最大絶対差 `1.1444091796875e-05` で合格した。検証用クラス名は `fixture-class-N` であり、ImageNet の意味ラベルとして提示していない。利用者が学習した ONNX での kit 実行結果は、ブラウザー書き出しの検証と合わせて別に記録する。

## 8. この記録単独では判定しない境界

- ブラウザーに独立実装した画像処理と元 native 出力の比較。
- ブラウザーの実勾配・重み更新と、書き出した完全 ONNX の独立再読込。
- ブラウザー書き出しモデルに対する Python と通常推論 kit の検証。
- 未使用の実画像での品質、実機資源量、中断・再開、プライバシー通信観測。
- D-FINE の位置推定を含む学習経路と、必要演算の forward/backward。

この文書の結果だけで上記を合格扱いにしない。その後に実施した分類のブラウザ学習・Python kit・通信観測は [P1a の実測記録](2026-10-24-browser-classification.md) にあり、残る未実施のゲートは [STATUS](../STATUS.md) で追跡する。
