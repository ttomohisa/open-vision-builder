# Open Vision Builder

A browser-based vision model builder for image classification and object detection, with local training and ONNX export.

**「Custom Vision が好きだった人向けの、もっとオープンな Vision Model Builder」**

画像の取り込み、アノテーション、追加学習、評価、モデルの持ち出しをブラウザで行うアプリを開発しています。学習にはユーザーの計算資源を使い、生成したモデルは Python、ユーザーの API サーバー、検証したエッジ端末へ持ち出す構成です。Browser Kitty とは独立したプロジェクトです。

## 現在できること

**P1a: 分類の技術検証を実装しました。製品の初期版と、分類・物体検出を含む P1 全体は未完成です。**

- 2〜8クラスの画像を取り込み、学習用と評価用に分ける。
- 事前学習済み **MobileNetV4 Conv Small** の特徴抽出部分を固定し、新しい分類ヘッドをブラウザで学習する。
- 学習に使わない画像で結果を確認し、完全な FP32 ONNX をブラウザで生成する。
- モデル、ラベル、入出力仕様、検証レポート、Python 推論例、モデルのライセンスを ZIP で持ち出す。

画像・ラベル・学習後の重みをサーバーへ送る経路はありません。アプリ、公開モデル、WASM の取得は同じサイトの静的ファイルへの GET です。アプリの操作で Python や変換サーバーを呼びません。

学習は現在 **WASM / CPU、分類ヘッドのみ**です。WebGPU、バックボーンの Fine-tuning、物体検出、実データでの品質、ピークメモリ、保存・再開は未対応または未検証です。生成サンプルの正解率は実画像の品質を示しません。ページを閉じると作業中のデータは失われるため、必要なモデルは先にダウンロードしてください。

[実測記録](docs/verification/2026-10-24-browser-classification.md)・[現在の状態](docs/STATUS.md)・[D-FINE の演算検証で見つかった問題](docs/verification/2026-10-24-dfine-probe.md)に、確認した範囲と残課題を記録しています。

## ローカルで試す

開発用には Node.js 24、Python 3.12 と公開重みを準備する環境を使います。**Python は開発者による公開モデルの準備と検証に使うものです。配信済みのアプリを使うユーザーには不要です。**

```bash
npm ci --ignore-scripts
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
npm run test:browser
.venv/bin/python scripts/predict.py --model artifacts/browser/model.onnx --verify-report artifacts/browser/report.json
```

ブラウザテストは `npm run build` と公開モデル準備後に実行します。実際の事前学習済み重みで学習・ダウンロードを行い、その出力を Python CPU でも確認します。サブパスを変える場合はビルドとブラウザテストの両方に同じ `OVB_BASE` を指定します。

`npm run probe:detector` は別の検証です。現在固定した jax-js の GridSample に既知の不一致があり、**終了コード1が未解決の検出ゲートを示します**。通常の合格テストへ置き換えません。

`dist/` は静的配信できます。GitHub Pages 用 CI・公開 workflow と Cloudflare 用の設定は [DEPLOYMENT](docs/DEPLOYMENT.md) を参照してください。Pages の初回設定と実際の公開確認は別工程です。公開確認前の URL はプレビューとして案内しません。

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
