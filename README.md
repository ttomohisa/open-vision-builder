# Open Vision Builder

A browser-based vision model builder for image classification and object detection, with local training and ONNX export.

**「Custom Vision が好きだった人向けの、もっとオープンな Vision Model Builder」**

画像の取り込み、アノテーション、追加学習、評価、モデルの持ち出しをブラウザで行うアプリを開発しています。学習にはユーザーの計算資源を使い、生成したモデルは Python、ユーザーの API サーバー、検証したエッジ端末へ持ち出す構成です。Browser Kitty とは独立したプロジェクトです。

## 現在できること

**P1a の分類画面を実装済み。P1b は D-FINE の限定した学習・完全 ONNX の基礎検証を追加しています。製品の初期版と P1 全体は未完成です。**

- 2〜8クラスの画像を取り込み、学習用と評価用に分ける。
- 事前学習済み **MobileNetV4 Conv Small** の特徴抽出部分を固定し、新しい分類ヘッドをブラウザで学習する。
- 学習に使わない画像で結果を確認し、完全な FP32 ONNX をブラウザで生成する。
- モデル、ラベル、入出力仕様、検証レポート、Python 推論例、モデルのライセンスを ZIP で持ち出す。

画像・ラベル・学習後の重みをサーバーへ送る経路はありません。アプリ、公開モデル、WASM の取得は同じサイトの静的ファイルへの GET です。アプリの操作で Python や変換サーバーを呼びません。

公開画面の学習は現在 **WASM / CPU、分類ヘッドのみ**です。開発用の検出試験では、COCO 学習済み D-FINE Nano の最終 bbox / class ヘッドを実更新し、完全 ONNX をブラウザと Python で照合しています。ただし、元実装と frozen prefix の中間特徴量の一致ゲートは **FAIL** です。検出モデルを画面から選択したり、Pages のモデルとして公開したりする段階にはしていません。

その後の[公開実画像10枚・12 stepの比較](docs/verification/2026-10-25-own-prefix-impact.md)でも、両実装の損失改善は事前条件に届きませんでした。完全 ONNX の fresh browser 一致は10/10画像、Python CPU 一致は8/10画像で、2画像の不一致を保持しています。検出器の製品採用を認定する結果ではありません。

独自クラスの検出、WebGPU、バックボーンの Fine-tuning、実データでの品質、ピークメモリ、保存・再開は未対応または未検証です。生成サンプルの正解率は実画像の品質を示しません。ページを閉じると作業中のデータは失われるため、必要なモデルは先にダウンロードしてください。

[分類の実測](docs/verification/2026-10-24-browser-classification.md)・[検出の部分結果と未解決ゲート](docs/verification/2026-10-24-dfine-foundation.md)・[現在の状態](docs/STATUS.md)に、確認した範囲と残課題を記録しています。

PR #4 後の追加調査では、一部入力で検出の最終出力にも不一致を観測しました。未 push の調査コードと生ログは作業環境の自動整理で失われたため、[調査メモ](docs/verification/2026-10-24-dfine-prefix-investigation.md)は会話記録からの復元であり、現時点の再現可能な合格証拠には使いません。その後、[矩形1画像の診断コードと新しい測定結果](docs/verification/2026-10-25-rectangle-reproduction.md)を保存しました。今回も中間特徴は不一致で、検出の採用は保留しています。

## ローカルで試す

開発用には Node.js 24、Python 3.12 と公開重みを準備する環境を使います。**Python は開発者による公開モデルの準備と検証に使うものです。配信済みのアプリを使うユーザーには不要です。**

```bash
npm ci --ignore-scripts
npm run prepare:deps
python3.12 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-models.txt
.venv/bin/python scripts/prepare_model.py
npm run build
npm run preview
```

表示されたローカル URL の `/open-vision-builder/` を開き、**「サンプルで試す」→「学習してONNXを作成」→「モデル一式をダウンロード」**と操作します。自分の画像では各クラスに学習用・評価用を1枚以上、合計64枚までを指定します。画像は PNG / JPEG / WebP、1枚20 MiB・1600万画素以下です。

公開モデルのソース revision と SHA-256 は固定しています。取得・一致検証が失敗した場合、準備処理は停止します。生成した重みやバイナリは git に含めません。

ZIP を展開した配備先では次のように推論できます。PyTorch やモデル変換は不要です。

```bash
pip install onnxruntime numpy Pillow
python predict.py --model model.onnx --image image.png --labels labels.json
```

## 検証と配信

```bash
npm test
.venv/bin/python -m unittest discover -s tests -p '*_test.py'
npx playwright install --with-deps chromium
npm run test:browser -- tests/browser.spec.ts tests/decode.browser.spec.ts
.venv/bin/python scripts/predict.py --model artifacts/browser/model.onnx --verify-report artifacts/browser/report.json
```

上記は分類のブラウザテストです。`npm run build` と公開モデル準備後に、実際の事前学習済み重みで学習・ダウンロードを行い、その出力を Python CPU でも確認します。引数なしの `npm run test:browser` は検出試験も含むため、下記の D-FINE 調査用成果物も先に準備します。サブパスを変える場合はビルドとブラウザテストの両方に同じ `OVB_BASE` を指定します。

`npm run prepare:deps` は、固定した `@jax-js/onnx` 配布物の全ファイル hash と MIT 表示を確認し、GridSample の隣接 index 計算を修正します。インストール時の自動スクリプトにはしていません。`npm run check:deps` は読み取り専用で適用結果を確認し、不明な版や変更済みの配布物は拒否します。`npm ci --ignore-scripts` のたびに準備をやり直します。

`npm run probe:detector` は修正済み GridSample の値・両勾配を検証し、現在は成功します。[元実装の失敗](docs/verification/2026-10-24-dfine-probe.md)と[修正の独立検証](docs/verification/2026-10-24-grid-sample-correction.md)を分けて記録しています。この演算の成功だけで検出器全体のゲートは合格にしません。

### 開発用 D-FINE 検証

検出の追加依存は分類と別の環境に入れます。準備処理は公開重みだけを扱い、各 revision・hash・形状・ライセンスを検査します。

```bash
python3.12 -m venv artifacts/p1b/.venv-detector
artifacts/p1b/.venv-detector/bin/python -m pip install -r scripts/requirements-detector.txt
artifacts/p1b/.venv-detector/bin/python scripts/prepare_detector.py
```

最後のコマンドは現在、**中間特徴量の不一致で終了コード1**になります。許容差は変更していません。全3入力・6出力の失敗レポートと調査用成果物を `artifacts/p1b/detector-assets/unqualified/` に残し、合格したモデルの出力先には公開しません。ダウンロード・hash・構造の失敗を、この既知の数値差として扱わないようにしています。

この失敗と独立に、同一特徴量を与えた最終ヘッドの forward・損失・全パラメータ勾配・SGD と、ブラウザで生成した完全モデルを検証できます。

```bash
artifacts/p1b/.venv-detector/bin/python scripts/generate_detector_training_reference.py --pretrained --cache-dir artifacts/p1b/detector-assets/source --output artifacts/p1b/training/pretrained-reference.json
node scripts/verify_detector_training.mjs --reference artifacts/p1b/training/pretrained-reference.json --report artifacts/p1b/training/pretrained-js-verification.json
npm run build
npm run test:browser
.venv/bin/python scripts/verify_detector.py --model artifacts/p1b/browser/model.onnx --report artifacts/p1b/browser/report.json --output artifacts/p1b/browser/python-verification.json
```

ブラウザ試験は調査用モデルをテスト専用のローカル URL で読むため、`public/` や `dist/` に D-FINE をコピーする必要はありません。部品検証の成功、検出の `detectorQualification: FAIL`、製品全体の未完了をレポートと ZIP に残します。80 COCO クラスを維持した最終段だけの学習で、独自クラス・元レシピ全体・実用精度の成立を意味しません。

`dist/` は静的配信できます。GitHub Pages 用 CI・公開 workflow と Cloudflare 用の設定は [DEPLOYMENT](docs/DEPLOYMENT.md) を参照してください。Pages の初回設定と実際の公開確認は別工程です。公開確認前の URL はプレビューとして案内しません。

### 分類の実画像品質・同条件 PyTorch 比較（開発者用）

[固定した設計](docs/superpowers/specs/2026-10-25-classification-real-images-design.md)と[実行計画](docs/superpowers/plans/2026-10-25-classification-real-images.md)に沿って、現在の分類画面の既定設定を公開写真で確認する準備を進めています。**実画像のモデル測定は NOT RUN、事前登録の公開と GitHub CI は未完了です。** [出典・利用条件・選択規則](docs/verification/2026-10-25-classification-data.md)を固定した5クラス60枚から、30枚を学習、20枚を評価に使い、調整用の10枚は今回は使用しません。

上記の依存関係・公開モデル・Chromium を準備した開発環境で、データ再現と推論しない事前確認を実行します。

```bash
.venv/bin/python scripts/prepare_classification_data.py
npm run build
node scripts/run_classification_real.mjs --check
```

`--check` は固定 bytes、bundle、Chromium 内の module 読込を確認し、画像のモデル推論や学習は行いません。この実画像比較について CI に追加するのは、データ準備と `--check` までです。

計画・入力 manifest・コードを PR に保存し、その SHA と最終検証を確認した後、一回だけ次の比較を実行します。

```bash
node scripts/run_classification_real.mjs
```

既定の40 full-batch SGD updates、learning rate 0.01 をブラウザと独立 PyTorch で比較し、完全 ONNX を別 browser context と Python で読み直します。実行上限は10分で、既存の `artifacts/p1e/run/` を上書きしません。品質・数値条件に届かない結果も保存し、結果に合わせた画像・設定の変更や自動再試行はしません。対象は canonical PNG、Small の固定 backbone と分類ヘッド、WASM の限定比較です。一般の JPEG decode、WebGPU、検出、製品リリースの認定は含めません。

### 検出の公開実画像での限定した学習影響比較（開発者用）

[事前登録した計画](docs/superpowers/plans/2026-10-25-dfine-own-prefix-impact.md)に従い、各 runtime 自身の frozen-prefix 特徴を使う12 step の比較を行うコードを用意しています。[画像・注釈・分割・利用条件](docs/verification/2026-10-25-impact-data.md)を固定し、同じ特徴・重みの対照が失敗した場合は停止します。

```bash
artifacts/p1b/.venv-detector/bin/python scripts/prepare_detector_impact_data.py
node scripts/run_detector_impact.mjs --check
node scripts/run_detector_impact.mjs
```

`--check` は固定ファイルと Chromium 内の module 読込までの事前確認です。推論・学習は行いません。引数なしの実験は最大15分で、既存の出力先 `artifacts/p1d/run/` を上書きしません。`status.json`、途中対照、全勾配・更新、比較、完全 ONNX を保存します。数値条件外の結果も保存し、構造・hash・実行エラーは非0終了します。実測の有無と結果は [STATUS](docs/STATUS.md)を確認してください。これらの開発用モデル・画像を Pages に公開したり、元の G1 FAIL を変更したりする処理ではありません。

## 製品として守る方針

- Cloudflare の静的ホスティング、運営側の API・DB・学習サーバーなし。
- 画像分類と、位置推定まで事前学習済みの物体検出モデルの追加学習。
- **初期版から、固定した重みと学習した重みを含む完全な ONNX 出力。**
- 商用利用・再配布条件を確認した正確なモデル配布物を使用する。
- モデルと利用開始用キットを提供し、生成モデルのホスティングはユーザーが行う。

分類の品質優先候補は MobileNetV4 Conv Medium、検出は D-FINE COCO Small、軽量検証候補は Conv Small / D-FINE Nano です。現在の分類ヘッド学習の成立を、D-FINE の学習成立として扱いません。

## 開発時に参照する文書

毎回 [AGENTS.md](AGENTS.md) から読み、会話の記憶だけに依存せず、変更した状態を同じ PR で更新します。

| 文書 | 役割 |
| --- | --- |
| [PRODUCT](docs/PRODUCT.md) | 合意済みの必須要件 |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | データ、学習、保存、ONNX、配信の境界 |
| [MODELS](docs/MODELS.md) | 固定したモデルと採用候補、出典、利用条件 |
| [VERIFICATION](docs/VERIFICATION.md) | 全体の合格条件と現在の部分結果 |
| [ROADMAP](docs/ROADMAP.md) | 導入順序 |
| [STATUS](docs/STATUS.md) | 実装済み・未検証・次の作業 |
| [設計判断](docs/decisions/README.md) | 選択の理由と変更履歴 |

専用ブランチから PR を提出し、マージはリポジトリ所有者が行います。

## ライセンス

アプリ本体のライセンスは未選定です。公開リポジトリであることを、特定の利用許諾が付与済みであるという意味には扱いません。今回固定した MobileNetV4 Small のモデルカードと実装は Apache-2.0 で、配信モデルと生成キットに LICENSE・NOTICE・モデルカードを同梱します。アプリ依存物の表示は静的配布物の `runtime/THIRD_PARTY_LICENSES.txt` に集約し、画面下部から参照できます。[依存物の出典と更新方法](third_party/README.md)も記録しています。詳細な出典と変更履歴は [MODELS](docs/MODELS.md) と [モデル準備の記録](docs/verification/2026-10-24-model-preparation.md)を参照してください。
