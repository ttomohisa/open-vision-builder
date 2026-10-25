# 現在の状態

更新日: 2026-10-25 JST。

## 段階

**P1a の分類はマージ済み。P1b は検出の最終ヘッド学習・完全 ONNX の基礎検証まで進行。検出の中間特徴量一致は FAIL、P1 全体と初期版は未完了です。**

合意・検証方針は [PR #1](https://github.com/ttomohisa/open-vision-builder/pull/1) と [PR #2](https://github.com/ttomohisa/open-vision-builder/pull/2) で記録しました。[分類の PR #3](https://github.com/ttomohisa/open-vision-builder/pull/3)、[検出基礎検証の PR #4](https://github.com/ttomohisa/open-vision-builder/pull/4)、[再現・局所診断の PR #5](https://github.com/ttomohisa/open-vision-builder/pull/5)、[検出の実画像比較の PR #6](https://github.com/ttomohisa/open-vision-builder/pull/6) はマージ済みです。PR #6 のマージは 2026-10-25 01:12 UTC（10:12 JST）。現在の比較元は `c088e211400e2db91e488858382fca39971cf41d`、作業ブランチは `test/classification-real-images` です。

## PR #4 後の作業と復元範囲

追加6入力で元実装と各 runtime 自身の特徴を使う学習を比較した際、候補契約は全件不合格でした。矩形入力では最終出力も不一致、ブラウザでは近い score の query 順序の入替を観測しました。学習の損失・勾配・更新値に部分一致の記録がありますが、検出器の合格にはしていません。

**追加の未 push コード・生レポート・ローカルコミットは、作業環境の自動整理で失われました。GitHub 上の PR #4 までの履歴は維持されています。** 現在の追加成果物は、会話の実行記録から復元した[調査メモ](verification/2026-10-24-dfine-prefix-investigation.md)と[再開方針](decisions/0004-dfine-investigation-checkpoint.md)です。追加の検証コードや、当時記録したテスト成功を、このブランチで実行・保存済みのものとして数えません。

PR #5 の初回コミットは文書のみでした。その後、[矩形1画像の最小再現](verification/2026-10-25-rectangle-reproduction.md)を再構築し、診断コードと小さい測定 JSON を保存しました。その測定では最終出力は許容差内、中間3出力は不一致でした。過去メモの最終出力不一致を完全に再現したものではありません。PR #6 で追加した実画像比較は後述します。

続く [decoder の切り分け](verification/2026-10-25-decoder-localization.md)では、最初の corner MLP が小さな入力差を増幅することを確認しました。同一 ONNX 入力を元 module に渡すと3 MLP は完全一致、3 decoder layer も許容差内でした。各 runtime 自身の入力では中間特徴の不一致が残り、G1 は FAIL のままです。

ユーザーの指示に基づく[方針点検](decisions/0005-bound-numerical-investigation.md)で、数値調査の自動延長を止めました。PR #5 は現在の再現・局所診断で区切り、予定していた decoder 全体 replay は保留します。次は学習・出力への影響を優先し、分類の実データ品質確認も独立して進める方針です。今回の点検で新しい学習・精度実験を実行したものではありません。

## 実装・検証の状態

### 分類の実画像比較: 事前登録の準備、モデル測定は NOT RUN

[設計と固定条件](superpowers/specs/2026-10-25-classification-real-images-design.md)・[実装計画](superpowers/plans/2026-10-25-classification-real-images.md)・[データと出典](verification/2026-10-25-classification-data.md)を用意しました。公開写真5クラス60枚をモデル出力を見る前に選択し、データ担当と統合担当の双方が全60枚を目視しました。各クラス6 train / 2 tuning reserved / 4 evaluation で、今回の学習・推論対象は train と evaluation の50枚です。

データの再現と固定 hash の検証に加え、レビュー修正後の JavaScript108/108件・Python112/112件・typecheck・build が成功しています。修正後の推論・学習を行わない Chromium `--check` も、固定した plan/data/model hash と対象50枚を確認して成功しました。GitHub CI は未完了です。**計画・コードの事前登録公開、実画像のモデル測定、結果の記録と判定はまだ行っていません。** 現段階を分類品質や完全 ONNX の実画像合格として扱いません。

予定は既存 UI の既定値である40 SGD updates・learning rate 0.01 と同条件の独立 PyTorch 比較、および別 browser context / Python による完全 ONNX 照合です。実行は事前登録後に一回、最大10分とし、条件未達も保存します。canonical PNG・Small の固定 backbone とヘッド学習・WASM に範囲を限り、一般の JPEG decode、WebGPU、検出、G4 全体・P1 全体・製品リリースの認定を含めません。

### PR #6: 公開実画像での限定比較を完了、条件未達を保存

[事前登録した計画](superpowers/plans/2026-10-25-dfine-own-prefix-impact.md)で、[公開実画像](verification/2026-10-25-impact-data.md)の4 train / 2 tuning / 4 evaluation、全58注釈、同一の12 SGD step を比較しました。[実測と次の判断](verification/2026-10-25-own-prefix-impact.md)・[比較 JSON](verification/evidence/2026-10-25-own-prefix-impact.json)を保存しています。整数 timeout の初回実行不具合は資料を残して修正し、データ・計画を変えず一回の完全実行を行いました。

0/6/12 step の同一特徴対照、全12 step の matching・最終 logits/boxes・損失・全勾配・更新値の部品比較は通りました。中間 `corners` は9/12 stepで不一致です。平均学習損失の改善は browser **0.004070%** / PyTorch **0.003928%** で、事前条件の1%に届きません。評価4画像の AP50 は両実装・更新前後で0.8463458012でしたが、有効な追加学習や実用品質を認定する結果ではありません。

完全 ONNX は fresh browser context で10/10画像が一致し、固定 initializer 576個と graph が不変、更新8テンソルの格納も一致しました。**Python CPU では6012/12667の2画像が元の最終出力条件外**です。6012は box の最大絶対差が0.3136631250あり、小さな丸め差と断定しません。学習効果・export の両方を条件未達として、追加測定をここで止めました。

評価画像は今回の追加学習に未使用という意味で、元 COCO checkpoint のモデル選定との独立は保証しません。画像単位の小規模比較であり、G4 全体や実用精度を認定しません。CI はデータ再現・単体テスト・ブラウザ起動の事前確認を行い、毎回の追加学習実験は自動実行しません。

| 項目 | 確認できた範囲 / 残課題 |
| --- | --- |
| 分類 UI | クラス定義、画像取込、明示的な学習/評価分割、実行、中断、評価、ZIP。矩形注釈・永続保存は未実装 |
| 分類モデル | MobileNetV4 Conv Small の正確な safetensors revision・ハッシュを固定。完全な公開モデルを準備・元 PyTorch と照合 |
| ブラウザ学習 | jax-js で新しい分類ヘッドを実更新。特徴抽出部分は固定。WASM 単一スレッドで検証 |
| 完全 ONNX | 固定重みと更新したヘッドを内包する単一 FP32 ONNX。別のブラウザ実行セッションで一致確認 |
| Python 利用 | ブラウザ生成モデルを ONNX Runtime CPU で実行し、公表 fixture の logits を比較。追加変換なし |
| GridSample | 固定した ESM/CJS の4行を hash 検証付きで修正。6手計算・12独立 PyTorch fixture の値・特徴/座標勾配を Node / browser WASM で比較 |
| 検出の限定学習 | D-FINE Nano の最終 pretrained bbox / class、計60,372パラメータ。元の2 step の部品検証に加え、実画像の各 runtime 自身の特徴で12 step を比較。両実装の損失改善は約0.004%で、1%基準に未達 |
| 検出の完全 ONNX | 更新した8テンソルを元の全グラフへ反映。従来の合成 fixture は照合済み。新しい実画像では fresh browser 10/10一致、Python CPU 8/10一致・2/10不一致。完全モデル生成と runtime 一致の未達を区別 |
| 検出の未解決 | **G1: 中間特徴量の一致は FAIL**。全3入力の元 PyTorch / ORT 比較を保存。最初の Conv の FP32 差と後段での増幅を確認したが、全演算の同等性を証明したものではない。検出 UI・独自クラス・完全 fine-tuning・精度は未実装/未検証 |
| 実データ品質 | 検出の公開10画像・同条件12 step比較は学習効果の基準未達で、独自タスク/G4全体を認定しない。分類は60枚のデータ準備・初回事前確認まで進行し、モデル測定は NOT RUN。Mediumは未検証 |
| 環境 | Linux x86_64 / Headless Chrome 156.0.8078.4 / WASM、非隔離サブパス。一般 PC、Edge、WebGPU、edge 端末は未認定 |
| 資源と継続性 | 経路の経過時間をレポート。ピークメモリ未測定。中断・古い結果の無効化は検証済み。保存・再開・バックアップは未実装 |
| データ非送信 | 公開生成データのブラウザ試験で外部通信・GET 以外の通信なし。運営 API / DB なし |
| GitHub Pages | PR #3 の実サイトで分類学習・ZIP・Python 一致を検証済み。新しい公開は commit と Actions を再確認。検出の未認定モデルは Pages に含めない |
| Cloudflare | 静的 dist と base 指定手順を用意。Cloudflare への実デプロイは未実施 |
| 利用条件 | 分類 Small と検証用 D-FINE Nano COCO の正確な重み・変換元・Apache 表示を固定し、成果物に同梱。Objects365 重みは使わない。アプリ本体のライセンスは未選定 |

## 証拠

- [ブラウザ分類・Python 一致の記録](verification/2026-10-24-browser-classification.md): 実行コマンド、環境、測定結果、制約。
- [分類の固定実画像と出典](verification/2026-10-25-classification-data.md)・[実画像比較の実装計画](superpowers/plans/2026-10-25-classification-real-images.md): 測定前に固定したデータ・条件、事前確認と未完了事項。モデル測定結果ではない。
- [公開モデル準備](verification/2026-10-24-model-preparation.md): revision、SHA-256、完全 ONNX、PyTorch と ORT の比較。
- [D-FINE に関係する演算 probe](verification/2026-10-24-dfine-probe.md): 固定した実装、再現値、失敗条件。
- [GridSample の修正](verification/2026-10-24-grid-sample-correction.md): 元の失敗と、固定 hash の修正後の独立比較。
- [D-FINE 基礎検証](verification/2026-10-24-dfine-foundation.md): 最終ヘッド・完全 ONNX の部分成功と、変更していない中間特徴ゲートの失敗。
- [D-FINE 実画像の学習・出力影響](verification/2026-10-25-own-prefix-impact.md): 同一特徴対照の成功、実経路の比較、学習効果と Python export の条件未達、打切り判断。
- [ADR 0002](decisions/0002-classification-spike.md): 分類の部分実装を先に確認し、検出ゲートを継続する理由。
- [分類の実装計画](superpowers/plans/2026-10-24-classification-spike.md): PR #3 の作業範囲。
- [ADR 0003](decisions/0003-dfine-foundation.md) と [検出の実装計画](superpowers/plans/2026-10-24-dfine-foundation.md): 部分検証と未解決条件を分けて進める判断。

学習対象がヘッドだけであること、サンプルが合成データであること、資源量・実用精度が未確認であることを UI とレポートにも表示します。上流の公開精度を本実装の達成精度として使いません。

## 次の作業

1. **分類の実画像品質と同条件 PyTorch 学習比較の事前登録を完成させる。** 固定した分割・レシピ・評価条件・コードを PR に保存し、その SHA と GitHub CI を確認してから、一回の測定と結果の保存へ進む。現在はモデル測定 NOT RUN。PR #6 の検出実験は条件未達を保存して終了しており、今回の分類作業で検出の数値診断を延長しない。
2. 検出は、PyTorch 自身も学習効果の1%基準を満たさなかったため、限定レシピ・学習量・データ適合性を再評価する。独自クラス、完全な training forward・補助/分布損失等の採用範囲は段階的に決め、事前学習済み位置推定を引き継ぐ。今回の12 step を過適合・実用品質の証拠にしない。
3. Python export の不一致は6012/12667の既知入力・保存済み比較を起点とする別課題。追加調査は [ADR 0005](decisions/0005-bound-numerical-investigation.md) の問い・修正仮説・結果別判断・予算が必要で、decoder 全体 replay や許容差調整を自動追加しない。元 G1 FAIL、検出 UI 非公開は維持する。
4. WebGPU と対象 PC の時間・ピークメモリ、中断・checkpoint 再開を検証し、P1 全体の採否を判断する。
5. [ROADMAP](ROADMAP.md) の P2 でプロジェクト保存・バックアップと注釈 UI を整える。リリース判定は [VERIFICATION](VERIFICATION.md) の全ゲートに従う。

ONNX 必須、ブラウザ学習、商用利用可能な重み、実際の検出器学習という要件は変更していません。
