# 検証計画とリリース判定

更新日: 2026-10-25 JST

状態: **分類の花写真・同条件 PyTorch・完全 ONNX の科学的比較は限定条件で PASS。ただし元 controller の通信検査 ERROR は保持する。GridSample は修正後 PASS、検出の中間特徴量 G1 は FAIL。全体のリリースゲートは未完了。** 本書の合格条件は維持し、部分結果・失敗・未実施を区別する。[分類の実画像結果](verification/2026-10-25-classification-real-images.md)、[従来の分類](verification/2026-10-24-browser-classification.md)、[モデル準備](verification/2026-10-24-model-preparation.md)、[GridSample 修正](verification/2026-10-24-grid-sample-correction.md)、[D-FINE 基礎検証](verification/2026-10-24-dfine-foundation.md) を参照する。

PR #4 後の[追加調査メモ](verification/2026-10-24-dfine-prefix-investigation.md)には最終出力の不一致も記録されています。ただし、未 push の検証コードと生ログが失われており、このメモでゲートの合格範囲を追加しません。G1 は FAIL のままです。その後の[矩形1画像の新しい実行記録](verification/2026-10-25-rectangle-reproduction.md)はコードと JSON を保存しており、中間特徴の不一致を再現しました。これも品質・ブラウザ学習・モデル全体の合格を追加するものではありません。

[ADR 0005](decisions/0005-bound-numerical-investigation.md) により、追加の decoder 分解を自動的に続けず、学習・出力への影響を優先します。これは検証作業の順序の変更であり、本書のゲート・既存の数値条件・G1 FAIL の変更ではありません。

## 1. 対象と変更できない条件

[AGENTS.md](../AGENTS.md) を作業規則とし、[PRODUCT.md](PRODUCT.md)、[ARCHITECTURE.md](ARCHITECTURE.md)、[MODELS.md](MODELS.md)、[STATUS.md](STATUS.md) と整合させる。
モデル名・公開重み・ライセンス・採用状態は MODELS、進捗は STATUS を参照する。

- 現代的な事前学習済み分類モデルと、位置推定まで事前学習済みの完全な検出器を対象とする。
- 学習対象はブラウザー内のユーザーデータ。画像、ラベル、特徴量、勾配、学習済み重みを運営側へ送信しない。
- 運営側のアカウントサービス・API・DB・学習サーバーを設けない。
- ユーザーのローカル Python による学習・変換を必須にしない。
- 公開ベースモデルのビルド時準備は許容する。ユーザーデータの学習・書き出しの代行には使わない。
- 初回リリースから、バックボーンと必要な全重みを含む単独の完全な FP32 ONNX を書き出す。
- 商用利用条件、実際の検出器ファインチューニング、上記のローカル処理条件を失敗時にも維持する。

現在の第一検証候補は MobileNetV4 Conv と D-FINE の COCO 学習済みモデルであり、モデル名自体を変更できない要件とはしない。候補を変更する場合も、上記の条件と以下の検証を適用する。

## 2. ゲート一覧

| ID | 判定対象 | 状態 |
| --- | --- | --- |
| G0 | ソース・重み・商用利用条件・変換履歴の固定 | PARTIAL: 分類 Small と検証用 Nano COCO の正確な safetensors・コード・Apache 表示・変換記録を固定。検出は製品未採用 |
| G1 | 元実装とブラウザーの forward 一致 | PARTIAL / detector FAIL: 分類の花写真50枚で元特徴・logitsの比較が条件内。検出の旧中間特徴の失敗と、own-path tuning 2画像の学習前後 logits/boxes 条件外は維持 |
| G2 | 学習パラメーターへの勾配と実更新 | PARTIAL: 分類ヘッド、修正 GridSample の値/両勾配、同一特徴での最終 pretrained bbox/class ヘッドの全勾配・SGD を確認。完全 detector backprop は未実施 |
| G3 | 少量データへの過適合と事前学習済み位置推定の更新 | PARTIAL: 分類の固定 backbone と head 学習は browser/native とも train30/30、CE約82.9474%改善。検出の位置重み更新は確認したが、実画像12 stepの両実装の損失改善は約0.004%で1%基準未達、検出の少量過適合は未認定 |
| G4 | 独自タスクの未使用データ評価と公平な PyTorch 比較 | PARTIAL evidence / NOT QUALIFIED: 花5クラスの同条件比較は両実装とも評価15/20、macro F1 0.7504761905で限定条件内。元 controller ERROR を別記。検出の学習改善未達、幅広い独自タスク・一般化品質・G4全体は未認定 |
| G5 | クラス数・空画像・画像形状の境界条件 | PARTIAL: 分類の入力/shape/class/取消。検出の空/単一/複数 target・矩形割当・disjoint GIoU 勾配・不正入力・取消を検証。独自クラスは未実施 |
| G6 | 学習後の完全 FP32 ONNX と独立した推論一致 | PARTIAL / real-image detector FAIL: 分類の完全 ONNX は花写真50枚の fresh browser / Python CPU 比較が各50/50条件内。検出は fresh browser 10/10一致、Python CPU 2/10不一致とG1 FAILを維持。edgeは未検証 |
| G7 | キャッシュ・非公開状態・隠れた通信への非依存 | PARTIAL: 分類と検出出力を別 CPU runtime で推論。kit はモデル追加取得・変換なし。全対応環境の認定は未完了 |
| G8 | 実機での時間・メモリー・中断・保存・再開 | PARTIAL: CI相当 Linux の経過時間と取消のみ。対象 PC・メモリ・再開は NOT RUN |
| G9 | データ非送信と再現可能な証拠の確認 | PARTIAL: 合成 fixture の E2E 通信観測・依存固定。花写真試験の元 controller は同一 origin の任意 build-info GET 一件で ERROR。科学的比較 PASS で通信検査の元失敗を消さず、製品全体の確認は未完了 |

全ゲートの証拠を揃えるまで、製品全体を「ブラウザーで学習可能」「ONNX 対応済み」「リリース可能」としない。部分実装について述べる場合は、分類 Small の固定特徴 + ヘッド、検出 Nano の既存80クラス最終ヘッド、WASM、公開 fixture、検証環境など実際の範囲を付記する。CI の部品検証成功は G1 FAIL を解消しない。厳格な検出準備コマンドの非0終了と、全比較の失敗レポートを保存する。

## 3. 実行前に固定する評価仕様

- ソース commit、チェックポイントの URL・ハッシュ、依存バージョン、モデルレシピ、変換処理を記録する。
- ブラウザー、OS、GPU、ドライバー、バックエンド、入力形状、バッチ、乱数条件を記録する。
- 前処理、座標系、クラス順、出力テンソル、後処理、評価指標を先に定義する。
- logits・特徴・boxes・scores・損失・勾配ごとに、比較方法と数値許容差を実行前に決める。
- ゼロ近傍・小さい box・離散的な選択結果は、同じ相対誤差だけで判定しない。
- 精度、時間、メモリーの合格条件も実行前に決め、結果に合わせて緩めない。
- 本書には未測定のベンチマーク値や根拠のない固定許容差を置かない。
- 少量過適合用データと未使用評価データを分離し、重複・近似重複・分割漏れを点検する。

## 4. G0–G1: 重みと forward の対応

- 公開重みが分類器だけ／バックボーンだけではなく、予定した完全なモデルの重みか確認する。
- 分類では分類出力、検出では特徴段・生の分類出力・位置推定出力を元実装と比較する。
- 同じ画像と同じ前処理で、元の PyTorch 実装、準備したグラフ、ブラウザーを比較する。
- クラス変更前にベースモデルの一致を確認し、変更後は同じ初期テンソルを両実装へ渡す。
- テンソル名、形状、軸順、共有パラメーター、固定／学習対象の対応表を保存する。
- Conv の重み配置、padding、resize、正規化、活性化、座標復元の差を確認する。
- BatchNorm の固定・融合、共有重みの複製、定数畳み込みは変換履歴と対応表に含める。
- 融合後の推論パラメーターを学習する場合は、元の train-mode 学習と異なるレシピとして扱う。
- 未対応演算、独自ドメイン、外部重み参照を列挙し、黙った置換や欠落を認めない。

## 5. G2–G3: 本当に検出器を学習できるか

### 実行系の前提

jax-js を最初の検証候補とする。TFJS でのモデル再構築は未選定の別経路であり、代替実証とはしない。
ONNX の推論成功や入力画像に対する勾配は、事前学習済み重みの更新を証明しない。
jax-js の ONNX ローダーは initializer を内部に保持するため、学習パラメーターを明示的な入力へ変換する設計を検証する。
公開 ModelProto を変更するだけで、作成済みの実行関数の重みが更新されると仮定しない。

### 学習グラフと損失

- D-FINE の公開 deploy グラフを、そのまま元の training forward と同一視しない。
- deploy で省略・変更される補助出力、denoising、反復 refinement、detach の位置を対応表にする。
- 採用する分類・位置・分布の損失、補助損失、教師信号を列挙し、元レシピとの差を明記する。
- Hungarian matching と IoU などの教師信号に必要な stop-gradient を明示する。
- matching の離散選択や GPU→CPU 読み戻しを、微分可能な計算と分けて検証する。
- GridSample の値側・座標側を含め、必要な演算の forward と backward を検証する。
- raw output を返すだけで後段実行が省略されると考えず、不要な後処理ノードを明示的に除く。

### 更新の証拠

- 学習対象テンソルを明示し、初期値が予定した公開事前学習済み重みに由来することを確認する。
- 分類では分類パラメーター、検出では事前学習済みの位置推定パラメーターへの勾配を確認する。
- 代表パラメーターの勾配を元実装または数値微分と比較し、optimizer の更新値も比較する。
- クラス射影だけでなく、bbox／位置分布を担う事前学習済み重みが実際に変化することを記録する。
- 固定対象が変化しないことと、更新が次回 forward・保存・書き出しへ反映されることを確認する。
- 少量の既知データで損失と予測 box の変化を確認する。入力勾配や分類正解だけでは合格にしない。
- NaN、Inf、ゼロ勾配、勾配の切断、誤った optimizer 状態の扱いを調べる。

## 6. G4–G5: 品質とデータ条件

- COCO の再評価だけでなく、対象用途の独自クラスを含む実データで検出・分類を評価する。
- 過適合用データを品質評価に流用せず、学習・調整に未使用の評価分割を使う。
- PyTorch 比較は同じ公開初期重み、分割、前処理、学習範囲、損失、optimizer、step 数・乱数条件を揃える。
- 主比較はブラウザー版と同じ学習レシピとし、元の公開レシピとの差を別に記録する。
- 元レシピの参考結果、学習前、ブラウザー学習後を混同せず、クラス別の失敗例も確認する。
- 分類指標、検出指標、未検出・誤検出を記録し、対象用途の事前合意条件で判定する。
- 1 クラス、複数クラス、事前学習時より多いクラス数、ラベル順序変更を試す。
- D-FINE の encoder／decoder の class projection、必要な埋め込み、shape 定数を確認する。
- クラス数変更と無関係な localization 重みを再初期化していないことを確認する。
- 空画像、複数物体、画像端、小物体、縦長・横長、異なる解像度を含める。
- 不正 box、空データ、未知ラベルは明示した入力規則に従って扱う。
- クラス追加・削除時に既存学習状態をどう継続／無効化するかを確認する。

## 7. G6–G7: 完全な ONNX と独立推論

- 学習済みの全推論ネットと全重みを単一の FP32 ONNX に格納し、部分ヘッドのみの出力を認めない。
- 更新した initializer、dtype、shape、クラス数、共有重み、BN 融合の対応を照合する。
- 学習用に追加したパラメーター入力を除き、推論時に要求する入力を明記する。
- optimizer・損失・matching などの学習専用要素と、推論に必要な演算を取り違えない。
- 生出力、前処理、後処理、座標復元、ラベル対応を deployment kit の仕様として固定する。
- 対応する opset・演算ドメイン・入力サイズを記録し、動的 shape の未検証範囲を約束しない。
- 保存直前のブラウザー出力と、別の新規 ORT Web セッションの出力を比較する。
- 同じ書き出しモデルを Python CPU の ONNX Runtime でも読み、同じ公開 fixture で比較する。
- 学習時の tensor、Worker、IndexedDB、OPFS、モデルキャッシュを引き継がずに推論する。
- 配布した kit のみを新しい環境で使い、モデルの追加取得や隠れたネットワーク依存がないことを確認する。
- Python は配布先での推論と開発時の公開 fixture 検証に用い、利用者の学習・変換要件にしない。
- ONNX が読めることだけで、全 edge 機器・量子化・各 accelerator 対応済みとはしない。

## 8. G8–G9: 実機、継続性、非送信

- 対象実機で初期化、初回／継続 step、評価、保存、書き出しの時間とメモリーを測る。
- GPU・CPU・タブ内メモリーと保存容量を区別し、device lost・OOM・未対応環境を確認する。
- 学習中の操作応答、取消、途中保存、タブ再開、checkpoint からの再開を確認する。
- checkpoint は必要な重み、optimizer、進捗、レシピ、乱数状態を保存する設計を検証する。
- 継続実行と再開後の結果を事前の許容条件で比較し、取消や例外で既存成果物を壊さないことを確認する。
- データ読込から書き出しまで通信を観測し、画像・ラベル・特徴・勾配・学習済み重みの送信がないことを確認する。
- エラー報告・分析 SDK・ログも対象とし、ユーザーデータや保存パスの意図しない収集を調べる。
- CI の元実装比較と Python 検証には、公開・利用許諾済みの固定 fixture のみを使う。
- ユーザーの書き出しモデルやデータを CI／運営サーバーへアップロードする経路を作らない。

## 9. 証拠と不合格時の判断

各ゲートは実行条件、コマンド／手順、入力・出力ハッシュ、比較結果、ログ、判定理由を記録する。
再現に必要な依存と公開 fixture を固定し、個人データを検証成果物へ混ぜない。
未実施は NOT RUN、失敗は失敗として残し、部分成功で全体を合格にしない。
失敗時は証拠、原因、選択肢、次の候補と再検証範囲を ADR に記録し、STATUS を更新する。
候補変更はモデルと実行系の両面で提案するが、ONNX・商用条件・バックエンド不要・実際の検出器学習を黙って緩めない。
サーバー学習への切替や分類ヘッドだけの学習を代替完了とせず、未達の制約を明示して設計を再判断する。

## 10. 一次資料

- [jax-js ONNX README](https://github.com/ekzhang/jax-js/blob/main/packages/onnx/README.md): 入力微分の例とローダーの位置付け。
- [jax-js ONNX 実装](https://github.com/ekzhang/jax-js/blob/main/packages/onnx/src/index.ts): initializer、入力、実行グラフの扱い。
- [jax-js ONNX 演算](https://github.com/ekzhang/jax-js/blob/main/packages/onnx/src/ops/convolution.ts): Conv と GridSample の実装範囲。
- [D-FINE decoder](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_decoder.py): クラス射影、位置推定、detach、deploy 変換。
- [D-FINE criterion](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_criterion.py): 損失と補助・教師信号。
- [D-FINE matcher](https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/matcher.py): no-grad の割当て。
- [D-FINE ONNX export](https://github.com/Peterande/D-FINE/blob/master/tools/deployment/export_onnx.py): deploy モデルと公開 export の入力・出力。

上記 URL の実装確認と本プロジェクトでの動作検証は別である。実行時には参照 commit を固定する。

2026-10-25 の [実画像 own-prefix 比較](verification/2026-10-25-own-prefix-impact.md)は、同一特徴対照の成功と、学習改善条件・Python export の未達を保存した。条件・画像・分割を結果へ合わせて変更せず一回の完全測定で終了した。旧 G1 FAIL と完全 ONNX 必須の製品要件は維持する。

同日の [分類実画像比較](verification/2026-10-25-classification-real-images.md)は、[PR #7](https://github.com/ttomohisa/open-vision-builder/pull/7) の事前登録 commit `261fcc70c4d795963349bb1f15d1fd816e4fe1f7` の条件を維持して一回だけ実行した。[比較証拠](verification/evidence/2026-10-25-classification-real-images.json)の科学的判定は `completed` / PASS、数値323/323件・品質条件が合格で、完全 ONNX は固定 initializer92個・node88個の不変と head6,405パラメーターを確認した。tuning10枚は使用していない。

**元の `status.json` は ERROR のまま保存する。** 科学的比較を保存した後、アプリが任意で読む同一 origin の `GET /open-vision-builder/build-info.json` 一件が strict harness の許可対象外となった。ローカル build にはこの CI 生成メタデータがない。`comparison.json` の PASS を controller 全体の正常終了や G9 全体の合格へ読み替えない。修正後はモデルを実行しない preflight で実際のローカル404応答を確認し、RED→GREEN を記録した。元の一回の結果・条件・エラーは変更していない。

[初回 CI 40151378784](https://github.com/ttomohisa/open-vision-builder/actions/runs/40151378784) では、ONNX 再生成値が歴史的な `28857aef…` となり、実測 artifact `eb823aa2…` を要求する分類 `--check` が失敗した。60画像の再現とモデルの既存生成 fixture 一致は成功したが、bytes の相違の原因や数値的同一性をこの事実だけから推測しない。元の実験 plan/hash/許容差と通常実行・`--check` の厳密な artifact 条件を維持する。CI は非測定モード `--check-app` を使い、artifact 自身の manifest/hash、同じ checkpoint/revision、dist の整合と App/helper 起動だけを確認する。`measurementModelMatches=false` の起動成功を実験の再現・G1/G4/G6 の追加合格として扱わない。最新コミットの CI 判定は [PR #7](https://github.com/ttomohisa/open-vision-builder/pull/7) に記録する。
