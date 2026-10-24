# D-FINE 最終ヘッド学習・完全 ONNX の基礎検証

実施日: 2026-10-24 UTC / JST。比較元: PR #3 マージ後の `037ef78e6e46c5191a11c7556c3127f2b28d6f15`。

## 判定

**限定した学習と完全 ONNX の部品検証は PASS。元実装と frozen prefix の中間特徴量の一致は FAIL。検出の製品採用・P1b 全体は未完了。**

この記録は COCO 学習済み D-FINE Nano の最終 bbox / class ヘッド、既存80クラス、単一スレッド WASM、手続き的な公開入力の結果である。独自クラス、元の全学習レシピ、少量過適合、実画像精度、WebGPU、対象 PC の資源量、保存・再開、edge 端末の結果ではない。判断は [ADR 0003](../decisions/0003-dfine-foundation.md)、製品全体の条件は [VERIFICATION](../VERIFICATION.md) に置く。

## 1. 正確なモデルと前後処理

| 項目 | 固定値 |
| --- | --- |
| 公開モデル | `ustc-community/dfine-nano-coco` / COCO-only |
| revision | `066438d3d8f0da137a37b38fdf3368fd4afceced` |
| `model.safetensors` SHA-256 | `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b` |
| 元ファイル | 15,278,996 bytes、644 tensors、モデル実パラメータ3,770,421 |
| PyTorch / Transformers | `2.6.0+cpu` / `4.56.0`、CPU FP32 eval |
| Transformers commit | `e7d351cebad5f6dcdd169b0c034fdee0a000e6a9` |
| 元 D-FINE の参照 commit | `956d1709314c2c6a4df6f34de232054578a7449f` |
| config SHA-256 | `7edf2d96d3deb10aa4ec920be0e8e444b617454545bb58093e32bd8e2b96fa2f` |

hash 検査済み safetensors から24個の明示的 shared-head alias を補い、strict load する。pickle / `.pth` は読み込まない。選んだ HF revision は Nano の K projection 変換修正を既に含む。比較する「元実装」は固定した Transformers の PyTorch を意味し、ネイティブ `.pth` との数値同一性は主張しない。

[固定モデルカード](https://huggingface.co/ustc-community/dfine-nano-coco/blob/066438d3d8f0da137a37b38fdf3368fd4afceced/README.md)は Apache-2.0 を宣言する。[D-FINE の COCO-only 重みの説明](https://github.com/Peterande/D-FINE/issues/357#issuecomment-5344034723)、両実装の LICENSE、モデルカード、変換来歴と下流変更の NOTICE を保持する。Objects365 重みは使用しない。全保存文書の URL / hash は `prepare_detector.py` と生成レポートにある。

入力は `images [1,3,640,640]`、FP32 NCHW、RGB/255、ImageNet 正規化なし。kit の画像入口は EXIF 適用・ICC 変換なし・白への浮動小数 alpha 合成、half-pixel bilinear の全画像伸縮である。数値比較自体は既に640×640の同一 FP32 入力を使い、全 codec の画素一致を主張しない。

出力は `logits [1,300,80]` と normalized `cxcywh` の `boxes [1,300,4]`。後処理は sigmoid、全 query/class pair の順位付け、元の縦横サイズへの xyxy 復元。softmax、NMS、座標 clip は行わない。kit の既定しきい値は0.3、最大100件であり、数値比較はしきい値前の全値を対象とする。

## 2. 完全グラフと調査用 prefix

| 成果物 | bytes | SHA-256 |
| --- | ---: | --- |
| `detector.onnx` | 16,161,249 | `4f282002c297af2835e038ea928e40cbcf56380321a914f85fd165e5aded8248` |
| `frozen-prefix.onnx` | 15,883,227 | `71f3591bcd5f33b6e23d152816e7eea7f81296771625bc932bcf0ccd6a7bdcf3` |
| `head-weights.json` | 1,829,485 | `5d69993cf89f6d9d23ddce01ae8033f761e6c4b40f99931bb85b741180b68953` |

IR8 / opset17、標準 ONNX domain、全重み内包、浮動小数 tensors は FP32。形状・index・mask に INT64 / BOOL を使う。完全グラフは3,563 nodes / 584 initializers / 52 operator types、prefix は3,438 / 572 / 51。全 inventory を保存する。

constant folding と deploy 再パラメータ化を無効にして export する。共有 alias と source-shaped initializer の値を検査し、最終学習対象8個が異なる initializer に対応することと、prefix に残らないことを確認する。未確定の静的出力 metadata は元実装で観測した shape を付け、全実行出力の形状を検査する。

元 export の FP64 scalar `1e-5` は、その全利用先が既存の FP32 Cast である場合だけ事前評価し、同じ FP32 bit の定数へ変換する。Cast 自体は残す。この限定した変換を検査し、他の FP64 tensor や外部データを黙って変換しない。

prefix は `bboxInput [1,300,128]`、`classInput [1,300,128]`、`residualCorners [1,300,132]`、`references [1,300,4]` を返す。最終 bbox MLP に必要な以前の hidden state との和、以前の corner residual、initial reference を元実装から取る。これらの raw feature gate が未合格である。

## 3. 固定した比較条件と失敗

手続き的な RGB pattern を phase 0 / 37 / 91 で生成する。

```text
r = (7*x + 3*y + phase) % 256
g = (2*x + 11*y + 3*phase) % 256
b = (13*x + 5*y + 7*phase) % 256
```

入力は FP32 RGB を FP32 255 で除算する。phase 0 の SHA-256 は `525561ded68fdb70dd9f2582caa85049908a294b1bc83e11fa6a521b5c76bbf5`。ユーザー画像や第三者の写真は含まない。

各有限要素について `abs(actual-reference) <= atol + rtol*abs(reference)` を検査する。許容差は初回実行前に固定し、失敗後も変更していない。

| 比較対象 | atol | rtol |
| --- | ---: | ---: |
| 元実装 ↔ ORT / browser の logits、bboxInput、classInput、residualCorners | 2e-4 | 1e-4 |
| 同 boxes / references、および完全 export の boxes | 2e-5 | 1e-4 |
| 同一特徴に対する pretrained head の forward / loss | 2e-4 | 2e-4 |
| 同 head の全パラメータ勾配 | 2e-4 | 2e-3 |
| 同 head の2 step後の全学習パラメータ | 5e-6 | 2e-5 |

Python ORT 1.20.1 CPU / single thread / default all optimization の結果:

| phase | logits | boxes | bboxInput | classInput | residualCorners | references |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | PASS | PASS | PASS | PASS | **FAIL** | PASS |
| 37 | PASS | PASS | **FAIL** | **FAIL** | **FAIL** | PASS |
| 91 | PASS | PASS | **FAIL** | **FAIL** | **FAIL** | PASS |

| phase | residualCorners 最大絶対誤差 | 最大許容比（1以下なら合格） |
| ---: | ---: | ---: |
| 0 | 0.002521515 | 4.624795 |
| 37 | 0.004999161 | 7.936587 |
| 91 | 0.003120422 | 8.150143 |

最終 logits の最大絶対誤差は phase 91 の0.0006108284、最大許容比0.715446で、相対項を含む元条件に合格する。boxes の全 phase の最大許容比は0.304481。最終出力の成功で特徴の失敗を取り消さない。

## 4. 失敗の診断

`scripts/diagnose_detector_parity.py` は元 Conv / BN / 以前の bbox MLP の中間出力を取り出して、同じ重みと入力を比較する。最初の Conv から差があり、最大差は3.4570694e-6。調べた最大差位置の27項内積では、FP64和1.35649086618に対し PyTorch=1.35648822784、ORT=1.35649168491で、双方の誤差は保守的な FP32 summation bound 2.8065393e-5 以下だった。

以前の bbox MLP を同一 ORT 入力で PyTorch 実行すると、両 MLP・全6 linear の出力は最大差0。前段の FP32 差が学習済み MLP で増幅する診断と整合する。ただし、全前段演算が同等であることの証明ではない。

ORT disabled/basic/extended/all、PyTorch MKLDNN on/off、固定 BN の定数事前計算では元ゲートを解消しなかった。FP64 prefix も元 FP32 参照に対して不合格で、30,501,903 bytes、ORT CPU の DOUBLE Conv 非対応となった。いずれの候補も製品変換へ加えていない。

別の phase 0 の影響調査では、空/単一/複数 target の matching・loss・parameter gradient は元の学習許容差を満たしたが、raw corner は58要素が不合格だった。これは**同じ1画像に対する3種類の target**で、3入力 phase 全ての学習同等性ではない。G1 FAIL は維持する。

## 5. 実際の pretrained ヘッド更新

学習対象は `model.decoder.bbox_embed.2.layers.{0,1,2}.{weight,bias}` と `model.decoder.class_embed.2.{weight,bias}` の計8 tensors。bbox は128→128→128→132、分類は128→80。50,052 localization + 10,320 classification parameters を更新する。前段と4個の最終 LQE tensors、33 bin の元 projection を維持する。

最終段の VFL + 5 L1 + 2 GIoU、Hungarian matching、detached IoU / VFL weights、FP32 SGD を独立 PyTorch で実装する。元 prefix を使った再構築 PyTorch head と元完全モデルの logits / boxes は最大差0。JS は同一特徴と全12 tensors を受け取り、空/単一/複数 target で全勾配を比較する。

| 同一元特徴を与えた Node WASM の実測 | 結果 |
| --- | ---: |
| 初期 logits / boxes の最大絶対誤差 | 2.8610e-6 / 5.9605e-8 |
| 全 parameter gradient の最大絶対誤差 | 6.3330e-8 |
| 2 step後の最大 parameter 誤差 | 5.8208e-11 |
| localization gradient L2 | 0.2968826653 |
| localization weight delta L2 | 5.9385754e-5 |
| classification weight delta L2 | 1.4424137e-6 |
| 実際に値が変わった学習 tensor | 8 / 8 |
| 値が変わった box 座標 | 845 / 1,200 |
| loss: 初期 → 2 step後 | 5.5658988953 → 5.5658817291 |

SGD は learning rate 1e-4、2 step。複数 target は COCO class index `[0,15]`、cxcywh `[.2,.3,.25,.35]` と `[.7,.65,.2,.3]`。これは任意に定めた経路検証用 target で、画像に対する実注釈ではない。小さな損失低下や box 変化は実更新の証拠であり、精度や過適合の証拠ではない。

jax 0.1.27 の境界も実際に失敗させて検査した。空 typed array の構築、disjoint box の `prod` 勾配 NaN は明示的な空配列と width×height で解消した。最終レビューでは FP32/JIT の大きな logit による softplus overflow、L1 の0点の勾配、ReLU の0点の勾配を修正した。安定した log-sum-exp と元 PyTorch と同じ subgradient を使い、固定 fixture / 許容差は変更していない。

## 6. ブラウザ生成物と独立 Python

Headless Chrome 156.0.8078.4 / Linux x86_64、`crossOriginIsolated=false`、ORT Web 1.30.0 / jax WASM、1 thread、`/open-vision-builder/pr-local/` の環境で実施した。

ブラウザが source / prefix / head weights / native fixture の hash を確認し、640×640 phase 0を生成する。ORT prefix からヘッドを2 step学習し、sourceの完全 ONNX の8 initializer を更新する。保存後の全グラフを新しい ORT session で開き、最終ヘッドの24,000 logits / 1,200 boxesと比較した。

| 実測 | 結果 |
| --- | --- |
| ブラウザ localization weight delta L2 | 5.9385881e-5 |
| ブラウザ classification weight delta L2 | 1.4424137e-6 |
| ブラウザ loss: 初期 → 2 step後 | 5.5658998489 → 5.5658826828 |
| 固定 LQE | 全値 bit-identical |
| fresh ORT Web の logits 最大差 | 1.9073486e-6 |
| fresh ORT Web の boxes 最大差 | 5.9604645e-8 |
| Python ORT CPU の logits 最大差 / 許容比 | 7.9154968e-5 / 0.107111 |
| Python ORT CPU の boxes 最大差 / 許容比 | 8.3446503e-6 / 0.092828 |
| ブラウザ経路の時間 | 約3.63秒。準備・推論・2更新・export照合を含み ZIP作成を除く。この環境だけの値 |
| 観測した通信 | 同一 origin/subpath の GET 20件、予期しない通信0、pageerror 0 |
| 検出 G1 | **FAIL**。browser residualCorners は123 / 39,600要素、最大差0.003608704、最大許容比3.786011 |

生成した完全モデルは **16,161,676 bytes**、SHA-256 **`e3dee0a35a478eaff36e45bddcce9686b920f5ad057737f7176a03b9846a88f9`**。外部重み・source checkpoint・学習中 tensor を借りずに実行する。Python verifier は独立に公開入力を再構築し、その hash、ONNX checker、全IO形状、全出力を検査する。

ブラウザが ZIP を作り、`model.onnx`、labels、manifest、report、ライセンス/NOTICE/カード、`detect.py`、README を入れる。`detectorQualification: FAIL` と調査用という表示を manifest/report/README/NOTICE へ引き継ぐ。コピーした実 ZIP を Python `-I` の別環境で実行し、NumPy 2.2.3 / Pillow 11.1.0 / ONNX Runtime 1.20.1 だけで推論した。PyTorch と Python ONNX package はその環境にない。追加変換・モデル取得なし。これは検出精度の認定ではない。

## 7. 再現手順と失敗の保存

[README](../../README.md) の依存準備後に実行する。分類と detector の Python venv は、huggingface-hub の固定版が異なるため分ける。

```bash
artifacts/p1b/.venv-detector/bin/python scripts/prepare_detector.py
```

現在は終了コード1。全18比較と完全な調査用 package を `artifacts/p1b/detector-assets/unqualified/` に保存する。通常出力先に usable model manifest は出さない。source/hash/shape/有限値/構造の失敗は診断モデルとして公開せず、atomic failure report だけを残す。以前の owned な生成物は無効化し、他のファイルがあるディレクトリは消さない。

```bash
artifacts/p1b/.venv-detector/bin/python scripts/diagnose_detector_parity.py --backend-matrix --fp64-prefix
artifacts/p1b/.venv-detector/bin/python scripts/generate_detector_training_reference.py --pretrained --cache-dir artifacts/p1b/detector-assets/source --output artifacts/p1b/training/pretrained-reference.json
node scripts/verify_detector_training.mjs --reference artifacts/p1b/training/pretrained-reference.json --report artifacts/p1b/training/pretrained-js-verification.json
OVB_BASE=/open-vision-builder/pr-local/ npm run build
OVB_BASE=/open-vision-builder/pr-local/ npm run test:browser
.venv/bin/python scripts/verify_detector.py --model artifacts/p1b/browser/model.onnx --report artifacts/p1b/browser/report.json --output artifacts/p1b/browser/python-verification.json
```

diagnose は原因の調査に成功すれば0終了するが、`qualifiesDetector:false` を記録する。全体ゲートを合格へ変更するコマンドではない。browser test はテスト用 route で ignored な調査ファイルを提供する。`public/` / `dist/` へモデルを入れず、Pages は分類画面のみを配布する。

最終確認: TypeScript / Vite build、JavaScript 回帰96件、Python41件、ブラウザ29件、独立 pretrained head、GridSample の PyTorch fixture再生成比較、分類・検出の Python ONNX 一致。別ディレクトリの clean `npm ci --ignore-scripts` で未修正拒否→明示 patch→読み取り検査→再適用の冪等性も確認した。

CI は厳格な準備の失敗を保持し、全比較と source の完全性・最終出力の成功を検査してから部品検証を続ける。job summary に **G1 FAIL** を表示し、回帰テストの成功でこれを隠さない。巨大な source/venv を保存せず、prepare/inventory/manifest、head report、browser ZIP/report、Python検証を artifact として残す。

主な証拠は `artifacts/p1b/detector-assets/{unqualified,parity-diagnosis.json}`、`artifacts/p1b/training/pretrained-js-verification.json`、`artifacts/p1b/browser/`。公開 fixture と source だけを使い、git にはモデル・ユーザーデータ・巨大な tensor を含めない。

## 8. 次の条件

中間特徴 G1、独自クラス、元の補助/分布損失を含む training forward、実データ少量過適合と未使用評価、対象PC/WebGPU/ピークメモリ、保存再開、edge互換性は未完了。数値条件を将来見直す場合も、今回の失敗は保存し、誤差予算の根拠と新しい未使用入力による事前定義の手順を先に記録する。観測済みの最大値へ条件を合わせない。
