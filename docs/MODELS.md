# モデル候補と採用基準

レビュー日: **2026-10-24 JST**

本書は ONNX を初期要件とする画像分類・物体検出のモデル選定方針を定める。
MobileNetV4 Conv Small は P1a の分類技術検証に限定して固定・変換・ブラウザ学習・ONNX 一致を確認した。D-FINE Nano COCO は P1b の最終ヘッド学習・完全 ONNX の調査用に固定したが、中間特徴量の一致は FAIL のため製品未採用。製品全体の品質・資源・検出学習の採用ゲートは未完了。

関連文書: [製品要件](PRODUCT.md) / [構成](ARCHITECTURE.md) / [検証計画](VERIFICATION.md) / [進捗](STATUS.md)

## 1. 初期候補

| タスク / プロファイル | 正確なモデル ID と重みの出典 | 用途 |
| --- | --- | --- |
| 分類 / 標準 | [`timm/mobilenetv4_conv_medium.e500_r256_in1k`](https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k) | 分類精度を優先する初期候補 |
| 分類 / 軽量 | [`timm/mobilenetv4_conv_small.e2400_r224_in1k`](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k) | メモリー・待ち時間を抑える候補 |
| 検出 / 標準 | [`ustc-community/dfine-small-coco`](https://huggingface.co/ustc-community/dfine-small-coco) | 学習済み検出器の精度を優先する候補 |
| 検出 / 軽量・最初の技術検証 | [`ustc-community/dfine-nano-coco`](https://huggingface.co/ustc-community/dfine-nano-coco) | 完全な検出器を小さい構成で検証する候補 |

分類候補は Google の MobileNetV4 アーキテクチャに対し、Ross Wightman が timm で学習した ImageNet-1k 重みである。
Google 配布の重みと同一であるとは扱わない。モデル ID の `conv` を含む構成まで固定する。

検出候補は COCO で学習した D-FINE の完全な検出器を起点とする。
複数物体のクラス・位置を予測するモデルであり、画像切り出しの分類を検出として扱わない。
独自クラスへの適応では分類出力の変更が必要になり得るが、学習済みの特徴抽出・位置推定を活用する。

### 公開値と未測定値

| 候補 | 公開パラメーター数 | 上流が報告する評価値 |
| --- | --- | --- |
| MobileNetV4 Conv Medium | 9.72M | ImageNet-1k top-1: 79.928%（256）、80.858%（320） |
| MobileNetV4 Conv Small | 3.77M | ImageNet-1k top-1: 73.756%（224）、74.616%（256） |
| D-FINE Small | 約 10M | COCO val2017 box AP: 48.5 |
| D-FINE Nano | 約 4M | COCO val2017 box AP: 42.8 |

分類値の出典は上記モデルカードの比較表。括弧内は評価入力サイズであり、製品設定の確定値ではない。
検出値の出典は [D-FINE 公式モデル表](https://github.com/Peterande/D-FINE) の COCO-only 行。
AP は IoU 0.50–0.95 の平均であり、分類 top-1 と比較する指標ではない。
これらは上流の評価結果であり、本プロジェクトによる再現結果ではない。
生成サンプルでの分類経路と経過時間は [実測記録](verification/2026-10-24-browser-classification.md) にある。実データでの品質、対象 PC の性能、ピークメモリーは未測定。
パラメーター数は ONNX 配布サイズや学習時メモリー量を意味しない。

## 2. ライセンスと商用再配布

上記 4 モデルのカードには、モデル単位で **Apache-2.0** が表示されている。
これは商用利用・再配布を検討する根拠であり、正確な配布成果物の権利確認が完了したことを意味しない。
コードのライセンスだけから、外部ダウンロードの重みまで同じ条件と推定しない。

| 対象 | 確認する一次資料 | 採用時に残すもの |
| --- | --- | --- |
| MobileNetV4 の実装 | [timm リポジトリー](https://github.com/huggingface/pytorch-image-models) の Apache-2.0 表記と LICENSE | 採用コミット、LICENSE、関連 NOTICE |
| D-FINE の実装 | [D-FINE リポジトリー](https://github.com/Peterande/D-FINE) の Apache-2.0 表記と LICENSE | 採用コミット、LICENSE、関連 NOTICE |
| 正確な重みファイル | 各モデルカード、ファイル一覧、配布条件 | 配布元、固定リビジョン、ファイル名、SHA-256、ライセンス根拠 |
| 変換・学習・実行依存物 | 実際に使う各プロジェクトの配布条件 | バージョン、ライセンス、成果物への同梱要件 |

静的サイトから配る重み・ONNX・学習用成果物も再配布対象として確認する。
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) の LICENSE、必要な著作権表示・帰属表示・NOTICE を保持し、変更したファイルで変更を明示する。
採用・変換・リリース担当が表示と同梱を確認する。公開前に未確認項目を残したまま採用済みにしない。
上流の表示と下流の表示が食い違う場合は、解消するまで採用を保留する。

### データセット由来の条件

D-FINE 公式 README は Objects365 由来の `*_obj365` / `*_obj2coco` 重みについて、
D-FINE のライセンスだけで商用利用可能と見なさないよう明記している。
初期候補は上記 COCO-only 重みに限定し、精度表の高い別チェックポイントへ無断で置き換えない。
COCO-only という名称自体を権利保証とせず、選択したファイルの学習来歴・配布条件を確認する。
ユーザーの学習画像・注釈の利用権は、事前学習モデルのライセンスとは別に扱う。

## 3. 出典と変換の連続性

D-FINE の公式実装はネイティブチェックポイントからの ONNX 書き出し手順を提供する。
一方、上記 D-FINE の Hugging Face 重みは Transformers 向けの変換済み成果物である。
両者を同じファイル・同じ変換経路と扱わず、実際に採用した経路を記録する。

1. 元の重みの配布元・リビジョン・ハッシュ・ライセンス根拠を確定する。
2. ネイティブ `.pth` と Hugging Face 重みを対応付ける場合、対応関係と数値的一致を確認する。
3. アーキテクチャ定義、キー対応、重み変換、前後処理の変更を再現可能な手順として保存する。
4. ONNX と学習用成果物を、元の重みと変換ツールのバージョンまで追跡可能にする。
5. 微調整・再書き出しした成果物にも、元モデルの出典・表示義務・変更履歴を引き継ぐ。

Hugging Face の重みを選んだ後に、便宜上別の `.pth` を使う変更もモデル変更として再確認する。
変換済みモデルが推論できることは、ブラウザーでそのパラメーターを学習できる証拠ではない。

## P1a で固定した配布物

| 項目 | 固定値 / 確認内容 |
| --- | --- |
| 重み | `timm/mobilenetv4_conv_small.e2400_r224_in1k` の `model.safetensors` |
| HF revision | `331fb803779522b685cf942e15f914fb6741c1eb` |
| 重み SHA-256 | `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4` |
| 実装 | timm 1.0.15 / commit `e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d` |
| 利用条件 | 固定したカードと timm は Apache-2.0。LICENSE、NOTICE、カード、変更表示を公開モデルと出力キットへ同梱 |
| 準備 ONNX | 15,069,036 bytes、IR8 / opset17、全 initializer 内包 |
| ONNX SHA-256 | `28857aef081dbba9e038c04cd8f2dabce28f654c66ffa47af1444179194f21d7` |
| 入出力 | `images [1,3,224,224]` → 元の `logits [1,1000]` と `features [1,1280]` |
| 学習対象 | 新しい線形分類 W `[1280,C]` / b `[C]`。それ以前の重み・統計は固定 |
| ユーザー出力 | 元分類ヘッドを除き、固定特徴抽出器と学習済みヘッドを内包した FP32 ONNX、logits `[1,C]` |

出典、変換処理、固定したライセンス本文の hash、元モデルとの比較は [モデル準備記録](verification/2026-10-24-model-preparation.md) を参照する。準備は checksum 検証後の safetensors のみを読み込み、pickle を使わない。ブラウザも配信 ONNX の checksum を検証する。固定済みの配布物の商用利用条件の確認と、製品の品質・全機能の完成は別に判定する。

## P1b で固定した検出の調査用成果物

| 項目 | 固定値 / 確認内容 |
| --- | --- |
| 重み | `ustc-community/dfine-nano-coco` の `model.safetensors`、COCO-only |
| HF revision | `066438d3d8f0da137a37b38fdf3368fd4afceced` |
| 重み SHA-256 / サイズ | `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b` / 15,278,996 bytes |
| 実行実装 | Transformers 4.56.0 / commit `e7d351cebad5f6dcdd169b0c034fdee0a000e6a9`、PyTorch 2.6.0 CPU |
| 変換元の来歴 | D-FINE reference commit `956d1709314c2c6a4df6f34de232054578a7449f`。HF 変換コード commit `be5169c704bae3d1d76d2890be5c72bcbc8c5de2` を保存 |
| Nano 変換修正 | 選んだ HF revision は `Fix weights (#1)` の merge。Nano attention K/V の既知修正を含む。別の checkpoint への差し替えはしていない |
| 利用条件 | 固定モデルカードは Apache-2.0。D-FINE と Transformers の LICENSE、モデルカード、変換・変更 NOTICE を保持。Objects365 重みは不使用 |
| ONNX | 全重み内包 FP32、opset17、640×640 / batch1。`detector.onnx` 16,161,249 bytes、`frozen-prefix.onnx` 15,883,227 bytes |
| 入出力 | `images [1,3,640,640]` RGB/255 → `logits [1,300,80]`、normalized cxcywh `boxes [1,300,4]` |
| 限定学習 | 最終 bbox MLP と class projection の8テンソル、60,372パラメータ。前段・LQE・既存80クラスは固定 |
| 採用判定 | 最終出力・限定ヘッド・完全 export は部分検証済み。中間特徴 G1 は FAIL。調査用ファイルは `artifacts/` のみで、Pages 配布・製品選択には含めない |

[固定したモデルカード](https://huggingface.co/ustc-community/dfine-nano-coco/blob/066438d3d8f0da137a37b38fdf3368fd4afceced/README.md)、[COCO-only 重みについてのメンテナの説明](https://github.com/Peterande/D-FINE/issues/357#issuecomment-5344034723)、[Nano 変換修正](https://github.com/huggingface/transformers/pull/40257)を来歴の根拠にする。HF safetensors と元の `.pth` の数値同一性は未検証であり、比較した元実装は固定した Transformers の PyTorch 実装を指す。

全 source 文書の hash、形状と initializer 対応、graph inventory、3入力の全比較を準備レポートへ保存する。型や hash の不一致は停止し、数値ゲートの失敗は明示した未認定の診断用ディレクトリだけに保存する。[D-FINE 基礎検証](verification/2026-10-24-dfine-foundation.md)と[ADR 0003](decisions/0003-dfine-foundation.md)を参照する。

## 4. モデルレジストリーの必須項目

以下は製品レジストリー全体に要求する項目である。P1a の公開 `model.json` と出力 `manifest.json` は schemaVersion=1 の限定スキーマで、この一覧を完全実装したものではない。
未確定の値は未確認として保持し、リビジョンやハッシュを推測で埋めない。

| 項目 | 記録内容 |
| --- | --- |
| `id`, `task`, `profile`, `status` | 正確な候補 ID、分類/検出、標準/軽量、採用段階 |
| `source` | 配布元 URL、固定リビジョン、ファイル名、SHA-256、取得日 |
| `code` | 実装元 URL、コミット、アーキテクチャ、必要な依存バージョン |
| `license` | コードと重みそれぞれの条件、根拠文書、商用再配布確認、LICENSE/NOTICE |
| `lineage` | 元チェックポイント、データセット来歴、変換元と変換先の対応 |
| `io` | 入力形状、色順、正規化、リサイズ/余白、出力形式、ラベル順序、後処理 |
| `artifacts` | ONNX opset、各成果物のハッシュ・サイズ、変換手順とツールのバージョン |
| `training` | 学習対象パラメーター、凍結範囲、損失、必要な学習成果物と生成方法 |
| `compatibility` | 実際に検証したブラウザー・実行プロバイダー・演算・制約 |
| `evidence` | 上流の指標と本プロジェクトの測定を分けた結果、検証日、採否理由 |

## 5. 採用ゲート

詳細な手順・合格条件は [VERIFICATION.md](VERIFICATION.md) に置く。
分類 P1a の部分結果は得られたが、製品全体として以下のゲートは未完了である。分類のみで満たした範囲を [VERIFICATION](VERIFICATION.md) と [STATUS](STATUS.md) に記載する。

| ゲート | 採用前に必要な証拠 |
| --- | --- |
| 出典・権利 | 正確なコードと重みの固定、商用利用・再配布条件、表示義務、変換来歴の確認 |
| 推論 | 選択した元モデルと ONNX の出力一致、前後処理、対象ブラウザーでの演算対応 |
| 学習 | 採用する学習範囲で勾配と更新が成立し、小規模データへの過学習試験が通ること |
| 検出の学習 | マッチング・分類損失・位置損失等を含む実際の検出学習経路の成立 |
| 再書き出し | 更新した重みを ONNX に反映し、再読込後の推論が更新後モデルと一致すること |
| 製品適合 | 実機での品質・待ち時間・ピークメモリー・中断復帰を評価し、対象範囲を明示すること |

上流の Python 学習手順や ONNX 推論エクスポーターは、ブラウザー内学習の成立を保証しない。
特に検出では完全な学習経路の検証が必要であり、ONNX 読み込み成功だけで実現可能と判断しない。
実際の採用状態と残課題は [STATUS.md](STATUS.md) に記録する。

## 6. 候補の変更

モデルや重みを「最新版」に自動追従させず、変更時はライセンス・変換・品質・実行条件を再確認する。
比較候補を追加する場合も、同じレジストリー項目と採用ゲートを適用する。
初期候補の制約を解消するための変更は、[製品要件](PRODUCT.md) と [構成](ARCHITECTURE.md) に反映する。
