# P1a: ブラウザ分類と完全 ONNX の検証

実施日: 2026-10-24 UTC。ブランチ: `feat/browser-classification-spike`。比較元: `64e9fe53579ca95d464ed85944f9bd3ef237336e`。

## 判定の範囲

**MobileNetV4 Conv Small の固定特徴 + 新しい分類ヘッドの学習から、完全 FP32 ONNX の生成・独立したブラウザ/Python 推論までを確認した。** これは分類 P1a の限定結果であり、[VERIFICATION](../VERIFICATION.md) の全ゲートや初期版の完成を意味しない。

実画像の品質、分類バックボーンの追加学習、D-FINE 全体の学習、WebGPU、対象 PC のピークメモリ、保存・再開、edge 機器は未検証。D-FINE に関係する [GridSample の不一致](2026-10-24-dfine-probe.md) は別の失敗結果として保持する。

## 実行条件と事前の合格条件

- Node.js 24.19.0、Python 3.12.14、Linux x86_64。
- Playwright 1.64.1、Chrome for Testing / Headless Chrome 156.0.8078.4。
- ORT Web 1.30.0 / jax-js 0.1.27 / optax 0.1.2、WASM 単一スレッド。GPU は使用・認定していない。
- 公開 [モデル準備](2026-10-24-model-preparation.md) の safetensors revision・hash、IR8 / opset17 の FP32 ONNX を使用。
- ネストした base `/open-vision-builder/pr-local/` で静的配信し、`crossOriginIsolated=false`、追加 COOP / COEP なし。
- 元モデル・書き出しモデル・Python 比較は、実装前の [計画](../superpowers/plans/2026-10-24-classification-spike.md) に定めた `abs(actual-expected) <= 2e-4 + 1e-4 * abs(expected)`。shape 不一致・NaN・Inf は失敗。
- E2E は生成した暖色/寒色の図形を学習8枚・評価4枚に分離し、40 epochs、learning rate 0.01、初期 head はゼロ。更新量が正、fresh export 比較が合格、ZIP が完全、評価 accuracy が0.5以上という経路の最低条件を事前設定。実用精度の条件ではない。

この環境では Playwright CDN が ZIP ではなく195 bytesのHTMLを返したため、同一版の [公式 Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/) の ZIP を取得し、CRC と実起動を確認して使用した。ブラウザの版を別物へ変えて試験を通す対応は行っていない。

## 再現するコマンド

開発用の依存・モデル準備は README の手順に従う。次はこの実測で用いた nested base の例。

```bash
npm test
.venv/bin/python -m unittest discover -s tests -p '*_test.py'
OVB_BASE=/open-vision-builder/pr-local/ npm run build
OVB_BASE=/open-vision-builder/pr-local/ npm run test:browser
.venv/bin/python scripts/predict.py --model artifacts/browser/model.onnx --verify-report artifacts/browser/report.json
```

ブラウザテストには Playwright Chromium の導入が必要。別の browser cache を使う場合だけ `PLAYWRIGHT_BROWSERS_PATH` を指定する。ここで用いた cache は task の作業環境内のもので、リポジトリや配布物に絶対パスを埋め込まない。

## 何を検証したか

- **勾配と学習:** `tests/head.test.ts` は独立に導いた2特徴/2クラスの1 stepの勾配・SGD更新、損失低下、有限値・shape・クラス制約、取消、繰返しとtensor解放を確認する。
- **完全モデル:** `tests/onnx.test.ts` は既知の凍結グラフと更新したヘッドを実 ORT で推論する。アプリは本物の MobileNetV4 の全必要固定重みを保持し、元1000クラス head を除去して更新した W/b を埋め込む。
- **ブラウザ操作:** サンプル→特徴抽出→実学習→fresh ONNX session→ZIPダウンロードまで操作する。入力変更後の古いダウンロード消去、取消、JS例外なし、外部 HTTP / GET以外の通信なしを確認する。
- **Python:** `onnx.checker.check_model(..., full_check=True)` と、別プロセスの ORT CPU で同じ公開入力と期待 logits を比較する。公開チェックポイントやブラウザ状態を読み込まない。
- **利用者 kit:** NumPy / Pillow / ORT だけの別 venv で、実際のブラウザ出力モデル、labels、公開 fixture PNG から推論する。torch / timm / onnx package が未導入であることを確認する。学習や変換は行わない。

ブラウザの `report.json` には公開合成パターンの検証用 tensor を入れる。学習に使用したユーザー画像を参照 fixture として混ぜたり、サーバーへ送ったりしない。ユーザーのラベル・評価結果・学習済みモデル自体は、本人がローカルへダウンロードする。

## 最終実測結果

| 検証 | 結果 |
| --- | --- |
| Build / TypeScript | PASS |
| JavaScript unit | 36 / 36 PASS |
| Python unittest | 11 / 11 PASS |
| 実ブラウザ | 27 / 27 PASS（全体操作2 + file decode25） |
| 学習範囲 | 1280次元固定特徴から2クラスのW/bを更新、40 epochs / lr0.01 |
| 最初 / 最後の損失 | `0.6931471824645996` / `0.16653935611248016`。各update前の値 |
| 学習重みの変化 L2 | `0.41799885570174894` |
| 合成画像の評価 | 学習8枚 / 評価4枚、4/4正解。実画像品質の判定には使わない |
| 経過時間 | `1820.9 ms`。run開始からexport検証まで。ZIP作成時間を含まない |
| native PyTorch → browser 最大絶対差 | `1.52587890625e-05` |
| 学習後 → fresh ORT Web 最大絶対差 | `1.1920928955078125e-07`、公開fixture + 評価画像4枚 |
| 学習後 → Python ORT CPU 最大絶対差 | `1.0132789611816406e-06`、公開fixture2 logits、許容外0 |
| ユーザー kit | 実ブラウザモデル + 公開PNGで同じPython logits。確率の和1.0。torch / timm / onnx packageなし |
| 単一 ONNX | `9955293` bytes、全固定重みと更新済みheadを内包 |
| ONNX SHA-256 | `78b7e3d6a5c6c186587cd66c80e7c971c10a4d11c45088d2f857ac921db30412` |
| 前処理 | 固定PNG / WebP / JPEGの25回帰。非zero-alpha画素byte一致、Python tensorとの差1e-6未満 |
| データ通信 | 全体E2Eで外部HTTP・GET以外のrequest・pageerrorは0 |
| 静的 asset | 25ファイル、最大15,069,036 bytes。すべて25 MiB未満 |

`artifacts/browser/` に完全ONNX、report、manifest、labels、画面を生成する。これらの大きい成果物はgitに含めず、CIの公開fixture用artifactへ保存する。この記録は上表の実行を特定し、別の環境や将来の実行で同じ時間・同じhashになるとは保証しない。

1280px幅と390px幅で表示を確認し、横方向のoverflowがないことも確認した。これはレスポンシブ表示の確認であり、モバイルでの学習性能の認定ではない。作業環境に日本語fontがなかったため、画面確認にGoogle公式のNoto Sans JPをローカル導入した。アプリの配布物には追加していない。

## 出力と来歴

ZIP は `model.onnx`、`labels.json`、`manifest.json`、`report.json`、`predict.py`、`README.txt`、`LICENSE.txt`、`NOTICE.txt`、`MODEL_CARD.md` を含む。クラス順、RGB/ImageNet正規化、half-pixel bilinear stretch、FP32 NCHW、出力 logits と softmax を明示する。モデルの出典、ソースの固定 revision/hash、分類ヘッドを変更したことも記録する。

公開モデル・WASM は同一 origin の静的 GET だけで取得する。画像・特徴・ラベル・勾配・学習後の重みを送信する API、分析 SDK、telemetry は設けない。元モデルの公開精度と、この stretch recipe の精度を同一視しない。

## レビューで修正した画像の入口

元の8-bit Canvas経路では、RGBA `[17,99,203,127]` のPNGが `[16,98,203,127]` に丸められ、白合成後の入力がPythonとずれた。RGBA配列だけの単体テストでは検出できないため、ファイルdecodeから最終tensorまでのブラウザ回帰を追加した。

PNG / lossless WebP は ImageDecoder の native packed RGB を `copyTo()` し、必要なB/R交換、行stride/offset、回転・反転を明示的に処理する。format変換やCanvasのpremultiplyを経由せず、ICC変換を無効化してPillowの扱いと揃える。非対応のalpha付き画素形式は明示的に拒否する。[WebCodecs仕様](https://www.w3.org/TR/2026/WD-webcodecs-20261019/)

JPEGと、native formatでopaqueと確認できたYUV画像は、`createImageBitmap` の ICC変換なし・EXIF適用とCanvasの読み出しを使う。ImageDecoderのYUV→RGBA変換では、同じ小さなカラーJPEGの4:4:4で最大1byte、4:2:0で最大40byteのPillow差を観測したため、この変換経路をJPEGには採用しない。不透明lossy WebPでも最大45byte差があったため、同じopaque専用bitmap経路へ分けた。修正した経路では両方の同fixtureの差は0になった。いずれも元ファイルを持つ固定fixtureの結果で、すべての画像・codec・ブラウザのpixel完全一致を保証するものではない。

テストは実production sourceを読み込み、PNG / lossless WebPのalpha `[0,1,2,127,128,254,255]`、RGB境界、ICC、EXIFを確認する。非zero-alphaのRGBはbyte一致、浮動小数点前処理は別実装のPython kitが生成した期待値に最大絶対差 `1e-6` 未満を要求する。ファイルサイズと画素数の上限もdecode前に確認する。

## 残る判定

この Linux 実行環境の短い合成試験の時間は、ユーザー PC のベンチマークではない。ピークメモリは測っておらず、レポートに `NOT MEASURED` と残す。WebGPU / Edge / Safari / Firefox / edge accelerator の対応を、この結果から推定しない。

GitHub Actions の実行と Pages の公開は、対象 PR の run / build-info.json によって別に確認する。今回のローカル HTTP の成功を、公開 URL の成功として記録しない。
