# 静的ビルドと GitHub Pages

このアプリは HTML、JavaScript、WASM、公開モデルを静的配信します。利用者の画像・学習・ONNX 出力はブラウザ内で処理します。以下の Python 処理は、開発者が固定した公開モデルと検証用データを準備・確認するためのものです。

workflow ファイルの追加だけでは Pages の初期設定や公開は完了しません。実際の公開状態は [STATUS](STATUS.md) と対象の Actions run を確認してください。

## CI と配布物

[CI workflow](../.github/workflows/ci.yml) の共通検証処理を全 PR で実行します。Pages 対象の同一リポジトリのブランチでは Pages workflow から一度だけ呼び出し、通常の CI job は重複を避けてスキップします。Node.js 24、Python 3.12 を使用し、次の順序で確認します。

```bash
npm ci --ignore-scripts
npm run prepare:deps
python -m pip install -r scripts/requirements-models.txt
python -m unittest discover -s tests -p '*_test.py'
python scripts/prepare_model.py
python scripts/generate_grid_sample_reference.py --check
npm run build
npm test
npx playwright install --with-deps chromium
npm run test:browser
python scripts/predict.py --model artifacts/browser/model.onnx --verify-report artifacts/browser/report.json
```

上記のビルド前に、CI は別の Python venv に `scripts/requirements-detector.txt` を入れ、`prepare_detector.py` と独立した pretrained head の参照生成・照合も実行します。[README の開発用 D-FINE 手順](../README.md#開発用-d-fine-検証)を参照してください。

検出の準備コマンドは現在、変更していない中間特徴量ゲートで終了コード1になります。CI はこの **G1 FAIL** を job summary の全比較表に表示し、全3入力6出力・source hash・最終 logits/boxes/references の合格・診断 manifest を検査してから、独立した部品検証を継続します。欠けたレポート、取得・構造・最終出力の失敗は CI 自体を失敗させます。`continue-on-error` はこの厳格な準備ステップに限定し、検出が合格したという表示には使いません。

既存ゲートの結果を検査した後、`reproduce_detector_rectangle.py` が既知の矩形1画像を CPU で診断します。通常の失敗不可ステップとして実行し、出典・構造・実行例外で CI を停止します。診断の終了0は測定の完了だけで、6出力の数値結果と `sourceParityPassed` を別の summary 表と JSON artifact に残します。この診断に `continue-on-error` は使いません。診断が完了しても検出器の資格判定は変更しません。

`npm run build` は依存修正の確認、ローカル配信用 runtime のコピー、TypeScript、Vite を含みます。ブラウザ検証が保存した分類・検出の完全 ONNX を Python CPU でも確認します。検出では `scripts/verify_detector.py --model artifacts/p1b/browser/model.onnx --report artifacts/p1b/browser/report.json` を実行します。CI が使うのは開発用の公開・生成 fixture で、利用者の画像・モデルを CI へ送る仕組みはありません。

未認定の検出モデルは `artifacts/p1b/detector-assets/unqualified/` に保存し、ブラウザ試験内の route だけで提供します。`dist/models/dfine-nano/` が存在すると配布確認を失敗させます。Pages に検出モデルや検証用画面は公開しません。

成功した run の `static-build-<commit>-<attempt>` artifact に `dist/` の内容を保存します。公開モデル、必要な利用条件の表示、WASM runtime を含みます。検証記録と診断情報は `browser-evidence-<attempt>` artifact に保存し、いずれも保持期間は7日です。モデルのバイナリや runtime は git にコミットしません。

CI は merge commit を検証します。Pages 用の同じ検証処理は、PR の head SHA を指定して実行します。`dist/build-info.json` に、実際に検証した `commit`、`pullRequest`、`basePath`、`workflowRun` を記録します。

## 初回だけ行う設定

公開前にリポジトリで次の2点を設定します。

1. **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にする。
2. **Settings → Environments → github-pages → Deployment branches and tags** で **Selected branches and tags** を選び、branch rule に `main` と対象の `refs/pull/N/merge` を追加する。`N` は実際の PR 番号。今後の信頼済み PR も許可する運用では `refs/pull/*/merge` を使えるが、workflow 側の同一リポジトリ・ブランチ制限は維持する。

設定後、**Settings → Secrets and variables → Actions → Variables** に repository variable **`PAGES_ENABLED` = `true`** を追加します。この変数は公開開始のスイッチで、Pages の初期設定を代行するものではありません。未設定の間も build・検証・artifact 保存は実行し、deploy job だけをスキップします。

`configure-pages` は `enablement: false` です。標準の `GITHUB_TOKEN` だけで Pages を新規有効化できるとは扱いません。既存の environment に承認ルールがある場合は、そのルールを維持して承認後に進めます。

設定を反映したら、該当 PR の **Actions → Pages → 対象 run → Re-run all jobs** を実行します。検証と artifact を再生成し、成功したものを公開します。新しいコミットへの更新でも同じ処理が走ります。run が実際に成功するまでは URL を公開済みのプレビューとして案内しません。

## 公開対象と URL の扱い

[Pages workflow](../.github/workflows/pages.yml) は次だけを対象にします。

| イベント | ソース | ビルド時の base path |
| --- | --- | --- |
| `main` への push | その push の commit | `/open-vision-builder/` |
| 同じリポジトリの `feat/browser-classification-spike` または `feat/dfine-training-foundation` から `main` への PR | PR の head SHA | `/open-vision-builder/pr-N/` |
| `main` 上の手動実行 | 選択した `main` の commit | `/open-vision-builder/` |

他のブランチと fork の PR は通常の CI のみを実行します。Pages 用 build も CI と同じ Node/Python、ブラウザ学習・ONNX 検証を通します。信頼済みブランチから `main` への PR では、通常の CI job をスキップして Pages 内の共通検証だけを実行します。`PAGES_ENABLED` が未設定でも、この検証は省略しません。

build job の権限は `contents: read` です。`pages: write` と `id-token: write` は、アプリのコードを checkout・実行しない deploy job にだけ付与します。Actions は公式リポジトリの commit SHA に固定しています。

**同時に公開するプレビューは1件です。** PR 公開では site root に案内ページを置き、`pr-N/` にアプリを置きます。次の Pages 公開でサイト全体を置き換えるため、前の PR パスは保持しません。`main` の公開は root のアプリへ戻します。複数 PR の永続的な並存や GitHub のネイティブな PR preview 機能は使用しません。

手動実行用の `workflow_dispatch` は用意していますが、新しい workflow がまだ default branch にない最初の PR で **Run workflow** が表示される前提にはしません。最初の PR は `pull_request` イベントと、その run の再実行を使います。ユーザーが PR をマージした後は `main` の push または `main` を指定した手動実行を使えます。

## 公開後の確認

Actions の deploy job が返した site URL を開き、次を確認してから PR へ動作確認 URL を記載します。

- `build-info.json` の `commit` が対象 PR の head SHA または対象の main commit と一致する。
- PR の場合は root のリンクから `pr-N/` が開く。アセット、Worker、WASM、公開モデルのリクエストがその base path で成功する。
- サンプルの学習、評価、ONNX 出力が動作する。preview ではサンプルデータを使用する。

GitHub Pages の custom header を前提にしません。非隔離環境で動く経路を検証し、`crossOriginIsolated` の有無と実行環境を結果に残します。最終的な Cloudflare 配信とは origin を分けます。`pr-N/` というパスの区別はデータのセキュリティ境界にはなりません。

## Cloudflare へ静的ファイルを配る場合

Cloudflare Pages の project root で配る場合は `OVB_BASE=/ npm run build` で作った `dist/` を配信します。モデル準備と検証は上記の開発用手順を先に実施します。Cloudflare 側にユーザーデータ用 API・DB・Python サーバーを設ける必要はありません。Direct Upload 等で、検証済みの静的成果物を配る運用が可能です。サブパスに配置する場合は、そのパスを `OVB_BASE` に指定して作り直します。

公開モデル・runtime・kit も `dist/` に含めます。Cloudflare Pages の1 assetあたり25 MiBという制限に対し、今回の公開 ONNX は15,069,036 bytes、WASM は約14 MBです。全 asset が制限内かビルドごとに確認してください。ブラウザが生成して直接ダウンロードするユーザーの ONNX は静的配信 asset にはしません。

今回の WASM 単一スレッド経路は COOP / COEP の追加ヘッダーを必要としません。WebGPU・マルチスレッドの対応を追加するときは、その実行条件を別途検証します。Cloudflare 本番への実デプロイは今回のローカル検証に含みません。

## 一次資料

- [GitHub Pages の公開元設定](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [Deployment branch rules と PR ref](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)
- [workflow の手動実行](https://docs.github.com/actions/managing-workflow-runs/manually-running-a-workflow)
- [configure-pages の入力と初回有効化条件](https://github.com/actions/configure-pages/blob/45bfe0192ca1faeb007ade9deae92b16b8254a0d/action.yml)
- [deploy-pages の権限と preview の制約](https://github.com/actions/deploy-pages/blob/368f82528645a54fb793d4d04e342629a3f51346/README.md)
- [Cloudflare Pages の制限](https://developers.cloudflare.com/pages/platform/limits/)
- [Cloudflare Pages Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/)
