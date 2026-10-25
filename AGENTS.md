# Open Vision Builder: 作業ガイド

このファイルは、人と AI が会話を引き継いでも、合意済みの目的と判断を参照して作業するための入口です。

## 作業開始時に読むもの

毎回、変更前に次を読みます。

1. [README.md](README.md): プロジェクトの目的と現在の段階。
2. [docs/PRODUCT.md](docs/PRODUCT.md): 確定した要件。
3. [docs/STATUS.md](docs/STATUS.md): 実装・検証の状態と次の作業。
4. [docs/ROADMAP.md](docs/ROADMAP.md): 今回の作業が属する段階。
5. 対象に関連する [ARCHITECTURE](docs/ARCHITECTURE.md)、[MODELS](docs/MODELS.md)、[VERIFICATION](docs/VERIFICATION.md)、[設計判断](docs/decisions/README.md)。

その後、実際のコード、変更差分、対象 PR のレビューを確認します。文書に「実装済み」とあっても、必要な動作確認は省略しません。新しいユーザー指示によって方針が変わった場合は、その指示を反映して関連文書を更新します。

## 守る製品要件

- Browser Kitty とは別のアプリ・リポジトリとして扱う。
- アプリは静的ホスティングで動作させ、運営側のアカウントサービス・API・DB・学習サーバーを設けない。
- ユーザーの画像・ラベル・勾配・学習済み重みを運営側へ送信しない。
- 画像分類と物体検出の一連のワークフローを対象にする。
- 学習後の重みを含む完全な ONNX 推論モデルを、初期版からブラウザで生成する。
- ユーザーに Python の変換環境を要求せず、ダウンロードした ONNX を Python 側でそのまま推論に使えるようにする。
- 事前学習済みのモデルを追加学習する。検出では学習済みの位置推定部分を引き継ぐ。
- 商用利用と必要な再配布条件を、コードだけでなく正確な重みの配布物について確認する。
- 生成モデルの配布用キットを用意し、そのホスティングはユーザーが行う。

開発用に公開重みを準備・変換したり、公開テストデータで参照値を作ったりする Python/CI 処理は、ユーザー向けの学習サーバーとは別です。

## 実装・検証の原則

- モデル名、採用候補、ライセンスの宣言、実機で確認した機能を混同しない。
- ONNX の推論成功を、モデルの重みが更新できる証拠にしない。
- 学習時と推論時の計算、前後処理、クラス数、正規化、重みの対応を明示する。
- 出力した ONNX を読み直し、ブラウザ内の最終モデルと Python の ONNX Runtime で結果を比較する。
- 精度・速度・メモリについて未測定の数値を書かない。上流のベンチマークには評価条件と出典を付ける。
- 合格条件を実行前に決め、失敗後に結果へ合わせて変更しない。条件の見直しには理由を記録する。
- 数値調査の追加は [ADR 0005](docs/decisions/0005-bound-numerical-investigation.md) に従い、問い・結果別の実装判断・実行予算・打切り条件を先に記録する。「さらに分解する」を自動的な次の作業にしない。
- 対応モデルの検証が失敗しても、ONNX 必須やブラウザ学習などの製品要件を黙って緩めない。[VERIFICATION](docs/VERIFICATION.md) に沿って原因と次の選択肢を記録する。
- ユーザーのデータ、認証情報、巨大なモデルのバイナリをリポジトリへ混入させない。参照用データは出典・利用条件を確認する。
- 実際に存在するビルドコマンドと、実行した確認を記載する。アプリ未実装の段階では文書の確認を行い、存在しないコマンドや成功していないテストを README・PR に記載しない。

## 現在の実行コマンドと検証範囲

準備・起動は [README](README.md)、配信は [DEPLOYMENT](docs/DEPLOYMENT.md) を参照します。

- `npm ci --ignore-scripts` の後に `npm run prepare:deps`。固定した jax-js ONNX の ESM/CJS/ライセンス hash を検証して4行の GridSample 修正を適用し、`check:deps` が未準備・未知の差分を拒否する。
- Python 3.12 の分類 venv に `scripts/requirements-models.txt`、別の `artifacts/p1b/.venv-detector` に `scripts/requirements-detector.txt` を導入する。依存版が異なるので同じ環境へ上書きしない。
- `python scripts/prepare_model.py`: 固定した公開重みを検証・変換し、ignored の `public/models/` に準備。ユーザーデータを使わない。
- `npm run build`: runtime / license の配置、TypeScript、Vite。
- `npm test` と `python -m unittest discover -s tests -p '*_test.py'`: 数値・export・Python 処理の検証。
- `npm run test:browser`: 準備済みモデルと build と Playwright Chromium が必要。実際のブラウザ生成物を `artifacts/browser/` に保存。
- `python scripts/predict.py --model artifacts/browser/model.onnx --verify-report artifacts/browser/report.json`: 完全モデルの Python CPU 一致確認。
- `npm run probe:detector`: 修正済み GridSample の独立値・勾配確認。現在は成功。修正前の失敗は元の記録に保持する。
- `python scripts/prepare_detector.py`: 検出の source / graph / 全3入力6出力を検証。中間特徴量の不一致では終了コード1、調査用ファイルを `artifacts/p1b/detector-assets/unqualified/` に保持する。許容差の変更や `|| true` での無条件通過は禁止。CI は失敗レポートの完全性と最終出力を検査してから部品検証を続け、G1 FAIL を明示する。
- `generate_detector_training_reference.py` / `verify_detector_training.mjs`: 同一元特徴に対する全ヘッド勾配・2 SGD step を比較。実行引数は README に記載。
- `python scripts/verify_detector.py --model artifacts/p1b/browser/model.onnx --report artifacts/p1b/browser/report.json`: 検出の完全 ONNX を独立 Python CPU で比較する。

P1a は分類の固定バックボーン + ヘッド更新です。P1b の部分成功は既存80クラスの最終 pretrained bbox/class ヘッドに限り、前段・LQE は固定です。中間特徴の G1 FAIL、独自クラス、全体 fine-tuning・実用精度・WebGPU・メモリ・保存再開は [STATUS](docs/STATUS.md) の残課題を引き継ぎます。検出の調査用モデルは `public/` / `dist/` に入れず、製品画面へ接続しません。モデルバイナリ・ユーザーデータ・生成 fixture の大きな tensor を git に含めません。

追加の `scripts/reproduce_detector_rectangle.py` は、既知の矩形1画像の CPU 診断です。終了0は測定完了であり、sourceParityPassed / 各出力の不一致と qualifiesDetector=false を保持します。新しい実行結果は [2026-10-25 の記録](docs/verification/2026-10-25-rectangle-reproduction.md)を参照し、失われた追加調査全体が復元済みとは扱いません。

同じ診断は [decoder の各段と同一入力 replay](docs/verification/2026-10-25-decoder-localization.md)も記録します。`references` は固定 anchor、各 layer の `reference_points` は更新後の消費座標です。局所 replay の成功で元の G1 を合格に変更しません。

2026-10-25 の方針点検で、予定していた decoder 全体の同一入口 replay は保留しました。PR #5 は保存済み診断の範囲で区切り、次は各実行系自身の特徴を使う学習・出力への影響を優先します。詳細は [ADR 0005](docs/decisions/0005-bound-numerical-investigation.md)。

## PR とマージ

- 作業ごとに専用ブランチを用い、`main` へ直接変更を push しない。
- 長い調査・実装はレビュー可能な小さな単位に分け、節目で専用ブランチへ保存する。未完成なら draft PR と未検証事項を残し、作業停止前に GitHub で保存済みの SHA を確認する。ローカル commit だけを永続保存と扱わない。
- PR #4 後の未 push 作業は自動整理で失われた。[復元メモ](docs/verification/2026-10-24-dfine-prefix-investigation.md)の過去の観測値を、現在のコード・生ログ・テスト成功の代用にしない。未保存の検証コードは再構築と再実行が必要。
- レビュー可能な単位で PR を作る。マージと自動マージの設定はユーザーが行う。
- PR には変更理由、実装した動作、確認した内容、未検証事項、必要なら動作確認 URL を記載する。
- 他の PR が未マージの場合は、依存関係と比較元を明記する。無関係な変更を含めない。
- GitHub Pages で確認できるアプリの導入は ROADMAP に従う。公開や動作を確認していない URL を「プレビュー」として示さない。

## 記録の更新

- 製品要件: `docs/PRODUCT.md`。
- 構成やデータの扱い: `docs/ARCHITECTURE.md`。
- モデル・重み・ライセンス: `docs/MODELS.md`。
- 合格条件・測定方法: `docs/VERIFICATION.md`。
- 実装済み、実測済み、未解決、次の作業: `docs/STATUS.md`。
- 導入順序の変更: `docs/ROADMAP.md`。
- 方針を選択・変更した理由: `docs/decisions/` に連番の記録。

該当する内容が変わる PR の中で更新します。文書だけの変更で ML 機能を実装済みにしたり、予定を実測結果として書いたりしないでください。
