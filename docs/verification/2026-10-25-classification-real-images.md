# 分類の実画像比較: 学習・品質・完全 ONNX

実施日: 2026-10-25 JST。対象: [PR #7](https://github.com/ttomohisa/open-vision-builder/pull/7)。

## 結論と次の判断

**固定した小規模試験の学習・品質・数値比較は PASS です。元の controller 実行は、任意のビルド情報取得を拒否したため ERROR のまま保存します。** 両者を一括して E2E PASS とはしません。

ブラウザと独立 PyTorch は、ともに学習30/30枚、今回の追加学習に使わなかった評価15/20枚を正解しました。学習 loss の40更新の履歴、最終 head、元モデルの特徴と出力、学習後の出力が元の比較条件内でした。ブラウザが生成した完全 ONNX は、学習 context を閉じた後の別 browser context と Python CPU で、全50枚が比較条件内でした。

この結果を現在の分類レシピを継続する根拠とし、次は分類データ・成果物のローカル保存/復元と、資源・中断時の扱いを進めます。20枚・5クラスの結果から一般的な精度、G4 全体、P1 全体、初期版完成を認定しません。検出の既存 FAIL と UI 非公開は維持します。

## 実行前の固定と一回の実行

- [事前登録コミット](https://github.com/ttomohisa/open-vision-builder/commit/261fcc70c4d795963349bb1f15d1fd816e4fe1f7): `261fcc70c4d795963349bb1f15d1fd816e4fe1f7`。PR は 01:37:41 UTC に作成。コミットには実画像のモデル測定 **NOT RUN** を記録した。
- 実行開始: `2026-10-25T01:38:24.623Z`。元 controller の停止まで42.7203秒。ブラウザ/参照の学習は各40更新、一回。モデルの再学習・再推論によるやり直しは行っていない。
- [設計](../superpowers/specs/2026-10-25-classification-real-images-design.md)、[実装計画](../superpowers/plans/2026-10-25-classification-real-images.md)、[出典・選択・変換・利用条件](2026-10-25-classification-data.md)を参照。
- plan SHA-256: `3b8f204d962dc9b52dfaa4b7f7f9844ea27161fe7c29aba3755b8a9ab4047b95`。
- dataset manifest SHA-256: `7d6885edda14c5e08b9a138d5f56ad9ab15b4d9ce3214b1d2befd9ceb5f80805`。

モデルは既存の MobileNetV4 Conv Small。元 checkpoint は `timm/mobilenetv4_conv_small.e2400_r224_in1k`、revision `331fb803779522b685cf942e15f914fb6741c1eb`、SHA-256 `7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4`。今回測定した準備 ONNX は `eb823aa24c3d3308ae8e3d65074634aa96800831bb238114207315a79d540e8a`、15,069,036 bytes。測定前の公開 main Pages のファイルとも byte 一致した。モデルの利用条件は [MODELS](../MODELS.md) を継続する。

花の5クラスを各6 train / 2 reserved tuning / 4 evaluation として固定した。実行対象は30 train +20 evaluation の50枚。調整用10枚は学習・推論・設定選択に使用していない。元 JPEG から固定手順で作った lossless RGBA PNG を、ブラウザと Python がそれぞれ decode / preprocess した。

既存 UI の40 full-batch SGD updates、learning rate0.01、平均 softmax cross entropy を使用。ゼロ初期化の W `[1280,5]` と b `[5]` の6,405パラメータだけを更新し、backbone/BatchNorm を固定した。shuffle、augmentation、momentum、weight decay はない。PyTorch は元 timm から自身の特徴を計算し、ブラウザの特徴を学習入力として流用していない。

## 学習と評価

| 指標 | ブラウザ | 独立 PyTorch | 事前条件 |
| --- | ---: | ---: | --- |
| 更新前 train CE | 1.6094379124 | 1.6094379124 | 同じゼロ初期 head |
| 更新40後 train CE | 0.2744511962 | 0.2744513284 | 改善率20%以上 |
| train CE 改善率 | 82.9473884% | 82.9473802% | PASS |
| train accuracy | 30/30 =100% | 30/30 =100% | 90%以上、PASS |
| evaluation accuracy | 15/20 =75% | 15/20 =75% | 70%以上、PASS |
| evaluation CE | 0.9820338102 | 0.9820337032 | 記録指標 |
| evaluation macro F1 | 0.7504761905 | 0.7504761905 | 実装間の差0.05以下、PASS |
| evaluation accuracy の実装間の差 | 0 | 0 | 0.05以下、PASS |

更新前のゼロ head は全クラス同点で、argmax の先頭クラスを選ぶため accuracy20%になる。事前学習済み1000クラス分類器の flower 精度を表す値ではない。

保存した loss40 は update40 の**直前**の `0.2818572819`。上表の最終 train CE は update40 後に別途計算したもので、最後の保存 loss と取り違えていない。

両実装の評価 confusion matrix は一致した。行が正解、列が予測。

| 正解 / 予測 | daisy | dandelion | roses | sunflowers | tulips | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| daisy | 3 | 0 | 0 | 0 | 1 | 75% |
| dandelion | 0 | 3 | 0 | 1 | 0 | 75% |
| roses | 0 | 0 | 3 | 0 | 1 | 75% |
| sunflowers | 2 | 0 | 0 | 2 | 0 | 50% |
| tulips | 0 | 0 | 0 | 0 | 4 | 100% |

両実装の誤分類は同じ5枚。結果を見て除外・再ラベル・再学習していない。

| 画像 ID | 正解 | 予測 |
| --- | --- | --- |
| daisy-459931395 | daisy | tulips |
| dandelion-477207005 | dandelion | sunflowers |
| roses-2491600761 | roses | tulips |
| sunflowers-1715303025 | sunflowers | daisy |
| sunflowers-164670176 | sunflowers | daisy |

撮影者名・画像 ID・JPEG/PNG/RGBA hash の重複を避けた小さい分割である。ImageNet 事前学習との画像重複・ドメイン独立は保証しない。sunflowers の2/4を含むクラス差を残し、75%だけで実用品質を保証しない。

## 数値と完全 ONNX

許容条件は事前の `abs(actual - expected) <= 2e-4 + 1e-4 * abs(expected)` のまま。323比較すべて PASS。独立前処理後の入力 tensor は50枚すべて最大絶対差0で、入力の `< 1e-6` 条件も満たした。

| 比較 | 件数 / 結果 | 最大絶対差 |
| --- | --- | ---: |
| 実 Worker の評価 scores と補助測定 | 20/20 PASS | 3.80491e-8 |
| source features | 50/50 PASS | 1.25766e-5 |
| source 1000-class logits | 50/50 PASS | 5.34058e-5 |
| 40更新の loss 履歴 | 1 trajectory PASS | 3.81470e-6 |
| 最終 head weight / bias | 両方 PASS | 1.34809e-7 / 1.72295e-8 |
| 学習後 browser / native logits | 50/50 PASS | 3.33786e-6 |
| fresh browser の完全 ONNX | 50/50 PASS | 4.76837e-7 |
| Python CPU の完全 ONNX | 50/50 PASS | 2.38419e-6 |

保存 head と特徴からの再計算50件も一致した。全比較の最大 tolerance ratio は約0.0601653（許容上限1）。5ポイント境界の浮動小数点表現を扱う実測前の修正は、今回の accuracy/F1 差がともに0なので判定に影響していない。

完全モデルは9,970,699 bytes、SHA-256 `cab71ffbd1387c62df00b8af08a74266e1a8f31598a2cf284f024a2cb4639830`。旧1000クラス head を除いた固定92 initializer / 88 node が不変であり、更新した6,405パラメータを埋め込んだ単一 FP32 ONNX と確認した。Python の元モデルの固定状態も不変だった。

実 App の file input → Worker → ZIP を通した。実 Worker の request、40 losses、評価 scores、summary と ZIP 内 report を照合した。補助 fixture の input/features/head-forward は同じ production 関数による追加測定であり、Worker 内部 tensor の直接記録ではない。head は export から抽出した。fresh context は学習 context の終了後に作成し、完全モデル・保存入力・runtime/helper だけを読み、元 backbone や写真の route を使っていない。

## 元 controller の ERROR と修正

独立比較が `completed / passed:true` を保存した後、controller は通信検査で終了コード1となった。唯一の rejected request は同じ loopback origin の `GET /open-vision-builder/build-info.json`。受理した124リクエストもすべて GET で、page error は0件だった。

App はこのファイルを起動時に任意取得する。CI はファイルを生成するが、通常のローカル build にはない。元 controller は存在するファイルだけを許可しており、元 `--check` は helper だけを開いたため、この App 起動時の取得を事前に検出できなかった。

修正では既存 metadata があればそのまま配信し、ない場合は**その正確なパスだけ**に本来の HTTP404 を返す。未知のパス・外部 origin・非 GET の拒否は維持した。事前確認に実 App 起動を加え、モデルを実行しない再現で修正前 FAIL → 修正後 PASS（metadata404）を確認した。

元 `status.json` の `status:error / passed:false` と通信記録は変更していない。学習・推論を再実行して成功した記録へ置き換えず、モデル比較 PASS と controller ERROR を分けて保存した。修正後の controller の完全実験は再実行していない。

## CI の範囲と異なる準備モデル

[事前登録コミットの CI](https://github.com/ttomohisa/open-vision-builder/actions/runs/40151378784)は、写真60枚をすべて固定 hash どおり再現した後、厳密なモデル artifact 確認で停止した。CI が再生成した ONNX は `28857aef081dbba9e038c04cd8f2dabce28f654c66ffa47af1444179194f21d7`（15,069,036 bytes）であり、実測した `eb823…` と異なる。既存の生成 fixture でのモデル準備比較は成功していたが、異なる bytes の原因や実画像での同等性は推測しない。

通常実行と `--check` は、今回の固定 artifact を引き続き厳密に要求する。CI 用 `--check-app` は、source checkpoint/revision、準備 manifest と実 bytes、dist の同一性を確認し、App 起動と helper import だけを行ってモデル実行前に終了する。実 artifact と測定 artifact の hash と一致/不一致を表示する。これは実画像測定の再現・モデル同等性の合格ではない。異なる artifact を通常実行で受理しないこと、全モードで破損や source 変更を拒否することを回帰テストで確認した。

JavaScript109件、Python112件、typecheck/build のローカル検証、厳密な no-inference preflight、App 起動だけの preflight が通った。独立したコード/保存結果レビューで、計画・データ・コード hash、実 Worker と ZIP の結合、保存 logits からの指標、凍結グラフ、元 ERROR の原因を確認した。最新コミットの CI 判定と URL は PR に記録する。

## 証拠、環境、限界

[比較 JSON](evidence/2026-10-25-classification-real-images.json)に323比較、50枚すべての予測/logits、指標、条件、元 controller と通信記録、実行コード hash、raw artifact69件の hash とサイズを保存した。大きな写真・入力 tensor・生 trace・モデル・ZIP・画面画像は ignored artifacts にあり、恒久保存済みとは扱わない。再現には固定モデル artifact の同一性確認が必要で、異なる環境の再変換だけで同じ bytes を得られるとは保証しない。

環境は Linux x86_64、Node24.22.1、Chromium156.0.8078.4、非隔離サブパス、WASM。Python3.12.14、torch2.6.0+cpu、timm1.0.15、ORT1.20.1、Pillow11.1.0 等の固定依存を使用した。Worker 計測区間は6.7742秒だったが、一般 PC の時間・ピークメモリ・再開性の認定には使わない。ローカルの画面保存では日本語フォントが不足しており、字形の見た目の合格資料には使用しない。操作・表示値は実 DOM と Worker/report で確認した。

canonical PNG、Small の固定 backbone + 分類 head、今回の5クラス、当該環境に限る。JPEG 全般、Medium、backbone fine-tuning、検出、WebGPU、edge、メモリ、保存・再開はこの測定の認定範囲に追加しない。運営 API/DB や画像アップロード経路は追加していない。
