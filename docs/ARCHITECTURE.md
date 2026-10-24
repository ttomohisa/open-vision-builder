# アーキテクチャ

更新日: 2026-10-24 JST

状態: 製品全体の設計方針と、実装した P1a の境界を記載する。分類の凍結特徴 + ヘッド学習、完全 ONNX、ブラウザ/Python 一致をローカル検証済み。物体検出、保存・再開、実デプロイなどの状態は [STATUS](STATUS.md) と [実測記録](verification/2026-10-24-browser-classification.md) を参照する。

## 1. 責務とデータの境界

Cloudflare の静的ホスティングからアプリを配信し、運営者の API・データベースを設けない。ユーザーの画像、ラベル、学習用分割、optimizer 状態、学習済み重みはブラウザー内で扱う。学習と ONNX 書き出しはユーザーのブラウザーで完結させる。

| 実行場所 | 責務 |
| --- | --- |
| Cloudflare の静的配信 | HTML、JavaScript、WASM、公開モデルパッケージの配信 |
| ユーザーのブラウザー | 画像取り込み、注釈、前処理、学習、評価、保存、ONNX 生成・検証 |
| 開発者の Python / CI | 公開チェックポイントの事前準備、参照出力の生成、ブラウザー出力の回帰検証 |
| ユーザー自身の配備先 | 書き出した ONNX を利用する Python、API、CPU edge、個別に認定するアクセラレーター |

API 配備用のキットを提供する場合、そのサーバー、認証、計算資源はユーザーが管理する。運営者側への画像・ラベル・学習結果のアップロード機能は設けない。

## 2. フロントエンドと実行基盤の候補

- フロントエンドは Vite + React + TypeScript を導入済み。正確な依存は package-lock.json に固定する。
- 初期の学習対象環境はデスクトップの Chrome / Edge と WebGPU を候補とする。対応バージョン、GPU、メモリー要件は技術検証で決める。
- メインスレッドは操作と描画を担当し、画像処理・学習・推論・ONNX 生成は専用 Worker で行う。job ID と Worker の破棄で中断後の応答を無効化し、入力変更で古い出力を破棄する。
- 学習エンジンは未選定。最初の技術検証候補は jax-js。TensorFlow.js を採用する場合は、対応する detector の学習可能な演算、損失、optimizer、重み変換を明示的に実装・検証する。
- P1a の分類は ONNX Runtime Web の単一スレッド WASM で固定特徴を抽出し、jax-js の自動微分と SGD で新しい分類ヘッドを更新する。完全 ONNX の独立推論にも ORT Web を使う。WebGPU は未認定で、成功実績を WASM から推定しない。

P1a の画像入口は PNG / WebP に ImageDecoder の native pixel copy、opaque JPEG と不透明なYUV画像に createImageBitmap + Canvas を使用し、ICC変換なし・EXIF適用を揃える。固定画像のブラウザ/Python回帰を持つが、未検証codecの全pixel一致を保証しない。

P1a は全画像を224×224へ half-pixel bilinear で伸縮し、ImageNet mean/std、FP32 NCHW を使う。元モデルの bicubic / center crop と異なる recipe で、上流ベンチマークを引き継ぐとはしない。1280次元の特徴を固定し、学習する W `[1280,C]` と b `[C]` を元の分類ヘッドと置換する。[ADR 0002](decisions/0002-classification-spike.md)

WebGPU は安全なコンテキストを必要とし、Worker 内でも adapter / device の取得を確認する。API の存在だけで対応済みとは判定しない。[WorkerNavigator.gpu](https://developer.mozilla.org/en-US/docs/Web/API/WorkerNavigator/gpu)

通常の Worker、WebGPU、WASM のマルチスレッドは別々に判定する。ONNX Runtime Web の WASM マルチスレッドは cross-origin isolation を必要とする。WebGPU 用 Worker は WASM proxy worker と分けて設計する。[ONNX Runtime の設定](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)

## 3. ローカル保存・バックアップ・再開

以下は製品化時の設計であり、P1a には未実装。現在はメモリ内の作業とモデル ZIP のダウンロードのみで、ページ終了後の復元はできない。

| 保存先 | 内容 |
| --- | --- |
| IndexedDB | プロジェクト、画像メタデータ、注釈、クラス定義、分割、学習履歴、モデル manifest |
| OPFS | 画像本体、サムネイル、重み、optimizer を含む checkpoint、大きな一時ファイル |
| Cache API | バージョンを固定したアプリ、ランタイム、ダウンロード済み公開モデル |
| プロジェクト書き出し | 画像、注釈、クラス対応、分割、設定、checkpoint を復元可能にまとめたバックアップ |

OPFS もブラウザーの容量制限を受け、サイトデータの消去で削除される。`estimate()` による容量表示、`persist()` の要求と結果表示、書き込み時の `QuotaExceededError` 処理を設計に含める。永続化要求の許可は保証されない。[OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)、[ストレージ方針](https://webkit.org/blog/14403/updates-to-storage-policy/)

checkpoint は一時ファイルの保存後に manifest を確定する方式を検証する。再開には重みだけでなく optimizer、step、乱数状態、分割・設定の識別情報が必要になる。再開時の再現性と中断耐性は検証対象であり、現時点では保証しない。

バックアップには schema version とファイルの checksum を持たせ、別ブラウザー・別端末への import を検証する。保存領域は origin に結び付くため、ドメイン移行にも export / import を用いる。[Storage Standard](https://storage.spec.whatwg.org/)

PWA / offline 対応を追加する場合は、アプリと必要モデルのキャッシュ完了を表示する。Service Worker に継続的な学習を委ねず、ページ終了・プロセス停止からの復旧を checkpoint で扱う。[offline と background 処理](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Offline_and_background_operation)

## 4. ONNX を必須にするモデルパッケージ

v1 の必須出力は、完全な単一ファイルの FP32 ONNX 推論モデルとする。更新した重みと凍結した事前学習重みの両方を含め、元のチェックポイントやブラウザーキャッシュへの依存を残さない。optimizer を含む再学習用 checkpoint は推論 artifact と分ける。

### 開発者による事前準備

公開チェックポイントについて、モデル系列・バージョン・hash を固定する。Python / CI で参照モデル、参照入力・出力、ブラウザー学習実装、ONNX グラフ生成 recipe、全 parameter の変換表を整える。開発者による準備は、ユーザーごとの学習・変換を受け付けるサービスではない。

### ユーザーごとのブラウザー処理

1. 公開モデルと recipe を読み込み、ユーザーのクラス構成で学習モデルを構築する。
2. 分類モデルまたは detector 自体をブラウザーで追加学習し、最終状態を確定する。
3. recipe と最終状態から推論グラフ・initializer を生成し、ONNX をローカルで直列化する。
4. 出力したバイト列を新しい ONNX Runtime Web session で読み、学習モデルの推論結果と比較する。
5. 検証結果と deployment manifest を添えてダウンロードする。

これは対応モデルごとの exporter であり、任意のモデルを変換する汎用 converter は前提にしない。ONNX の protobuf schema と tensor の little-endian 表現に従う。[ONNX schema](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)

## 5. グラフと重みの変換規則

| 対象 | 必須規則 |
| --- | --- |
| parameter 対応 | 元の名前・shape・dtype、宛先 initializer、変換を記録。未対応、形状不一致、非有限値を検出して書き出しを停止する。 |
| convolution | 学習側が HWIO の標準畳み込みなら ONNX の OIHW へ変換する。NHWC / NCHW の activation 変換とは分け、grouped / depthwise も個別に検証する。 |
| BatchNorm | train / eval の状態と統計更新方針を明示する。最終の scale・bias・running mean・running variance・epsilon を推論へ反映する。 |
| fusion | Conv + BN を融合するなら fine-tune 後の値で再計算する。事前に融合した重みへ一部だけを上書きしない。 |
| クラス数変更 | head の initializer、内部 channel、reshape / split / slice 定数、型・shape、出力定義を recipe から再生成する。出力 shape の変更だけで済ませない。 |
| graph 最適化 | 対応表が追跡できる決定的な処理に限定し、変換前後の出力を比較する。 |

ONNX Conv の重みは `[out_channels, in_channels/group, kernel_height, kernel_width]`。D-FINE のクラス射影や denoising 用埋め込みなど、クラス数に依存する箇所はモデル固有の構築規則で追跡する。[ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html)、[D-FINE decoder](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_decoder.py)

BN 融合は eval 状態で `s = gamma / sqrt(variance + epsilon)`、`W' = s * W`、`b' = s * (b - mean) + beta` とする。明示的な BN node を保つ FP32 export も候補とする。[PyTorch の融合処理](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html)

## 6. deployment manifest と互換性

| 項目 | 記録する契約 |
| --- | --- |
| 入力 | tensor 名、shape、dtype、batch、画像方向、RGB / BGR、channel layout |
| 前処理 | resize の補間、letterbox の丸め・padding 値、正規化、元画像座標への逆変換 |
| 出力 | tensor 名・layout、box 形式と座標系、class ID の順序、score の意味 |
| 後処理 | decode、confidence threshold、NMS の有無・場所・方式・IoU threshold、最大検出数 |
| 来歴 | 公開重み hash、recipe / 学習 engine の版、クラス定義、分割識別子、設定、評価結果 |
| 実行条件 | ONNX IR / opset、必要 operator / dtype、ランタイム版、検証済み環境、全ファイルの hash |

NMS の有無を含め、選んだモデルの生出力と後処理の契約を固定する。検出器すべてに同じ後処理を付け足さない。[D-FINE の公式 ONNX exporter](https://github.com/Peterande/D-FINE/blob/master/tools/deployment/export_onnx.py)

初期の互換性目標は ONNX Runtime Python の CPU 実行と、名前・構成を特定した CPU edge profile とする。IR / opset、shape、operator、dtype を対象 runtime で認定する。opset 番号だけを書き換えて互換化しない。[ONNX Runtime 互換性](https://onnxruntime.ai/docs/reference/compatibility.html)

NPU は個別 profile とし、static shape、operator 対応、コンパイル・calibration の条件を確認する。INT8 / FP16 は FP32 基準に対する精度・速度・メモリーの実測後に追加する。INT8 の高速化は保証しない。[QNN の条件](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)、[量子化の注意点](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## 7. 静的ホスティングと preview

Cloudflare Pages は 1 asset あたり 25 MiB を上限とする。公開モデルだけでなく WASM も確認する。ブラウザーが生成してユーザーへ直接ダウンロードする ONNX には、この配信 asset 制限を適用しない。[Pages の制限](https://developers.cloudflare.com/pages/platform/limits/)

静的 deployment kit の任意の派生成果物として、ONNX external data の `location` / `offset` / `length` で重みを別ファイルにする構成を検証する。必須の単一 FP32 ONNX をこの形式で置き換えない。各 tensor の全 byte 範囲を 1 shard に収め、全 shard を同梱する。1 tensor 自体が上限を超える場合、単純なファイル分割では解決しない。検証済みの graph 分割、別 precision / model、配信先変更を別途判断する。[ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html)

Cloudflare の `_headers` で COOP / COEP を設定する案を検証する。同一 origin の runtime / model 配信を基本とし、cross-origin resource の読み込み条件も確認する。[Pages headers](https://developers.cloudflare.com/pages/configuration/headers/)、[cross-origin isolation](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated)

GitHub Pages の基本 base path は `/open-vision-builder/`。asset、Worker、WASM、モデル URL と Service Worker scope をこの subpath に対応させる。PR ごとの subpath を使う場合は build 時の base も追加調整する。

信頼済みの同一リポジトリ PR を対象に、明示的な workflow で preview 用 subpath に公開する案を検証する。GitHub Pages の標準機能による自動 PR preview は前提にしない。workflow は実装済みで、初回の Pages 設定と公開の確認手順は [DEPLOYMENT](DEPLOYMENT.md) に記録する。公開の成否は Actions と対象コミットの build-info.json で確認する。[公式 deploy action](https://github.com/actions/deploy-pages)、[subpath preview の実装例](https://github.com/rossjrw/pr-preview-action)

GitHub Pages では custom header に依存せず、非隔離環境での実行経路を検証する。production と preview は origin を分け、preview はサンプルデータで確認する。パス名や IndexedDB 名の区別はセキュリティ境界にはならない。

## 8. 実装を進めるための検証境界

最初に「ブラウザー fine-tune → 完全な ONNX のローカル生成 → 新しい browser session → Python CPU」の一致を確認する。developer CI はブラウザー生成 fixture に `onnx.checker.check_model(..., full_check=True)` と実推論を実行する。ユーザーごとの export に Python や運営者の変換 endpoint を要求しない。[ONNX checker](https://onnx.ai/onnx/api/checker.html)

構造検証に加え、クラス数の境界、空の検出、各種画像比率、前後処理、学習後の parameter 反映、移動後の artifact 読み込みを確認する。判定基準と実施記録は [VERIFICATION.md](VERIFICATION.md) に集約する。分類の部分結果は [実測記録](verification/2026-10-24-browser-classification.md)、検出の演算不一致は [probe 記録](verification/2026-10-24-dfine-probe.md) にある。未実施の項目を合格としない。
