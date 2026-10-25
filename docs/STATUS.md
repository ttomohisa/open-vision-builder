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

### PR #7: 分類の科学的比較は PASS、元 controller の ERROR を保持

[設計と固定条件](superpowers/specs/2026-10-25-classification-real-images-design.md)・[実装計画](superpowers/plans/2026-10-25-classification-real-images.md)・[データと出典](verification/2026-10-25-classification-data.md)を [PR #7](https://github.com/ttomohisa/open-vision-builder/pull/7)、commit `261fcc70c4d795963349bb1f15d1fd816e4fe1f7` で公開してから、一回の科学的比較を行いました。画像・分割・40 SGD updates・learning rate 0.01・判定条件は変更していません。選択60枚のうち train 30枚と evaluation 20枚を使い、tuning reserved 10枚は未使用です。

[結果](verification/2026-10-25-classification-real-images.md)・[比較証拠](verification/evidence/2026-10-25-classification-real-images.json)では、ブラウザと独立 PyTorch はともに **train 30/30、evaluation 15/20（75%）、evaluation macro F1 0.7504761905** でした。train CE は1.6094379124から browser 0.2744511962 / native 0.2744513284へ低下し、改善率はそれぞれ82.9473884% / 82.9473802%です。入力50件は同一で、数値比較323/323件、完全 ONNX の fresh browser / Python 比較は各50/50画像で事前条件内でした。

**科学的比較の `comparison.json` は `completed` / `passed=true` / `numericPassed=true` / `qualityPassed=true` ですが、元 controller の `status.json` は `error` / `passed=false` です。** 比較保存後の通信検査が、アプリの任意メタデータ取得 `GET /open-vision-builder/build-info.json` を未許可と判定しました。unexpected request はこの同一 origin の一件で、page error はありません。このファイルは CI が生成し、通常のローカル build には含まれません。元の ERROR と通信記録を保持し、controller 全体の正常終了とは扱いません。

この一回で実験を終了しました。任意メタデータの扱いは strict harness と推論しない事前確認で修正し、実際のローカル404応答を通す RED→GREEN を確認しています。モデル学習・推論は再実行していません。事前の JavaScript108/108件・Python112/112件・typecheck・build は成功しました。修正後は JavaScript109/109件・typecheck・厳密な `--check` と `--check-app` が成功し、独立した結果・修正レビューも完了しました。これらは元 controller の ERROR を取り消す証拠ではありません。

[初回 CI 40151378784](https://github.com/ttomohisa/open-vision-builder/actions/runs/40151378784) は、再生成 ONNX が過去記録と同じ `28857aef081dbba9e038c04cd8f2dabce28f654c66ffa47af1444179194f21d7`（15,069,036 bytes）となり、実測・配信中の `eb823aa24c3d3308ae8e3d65074634aa96800831bb238114207315a79d540e8a` に固定した分類 `--check` で失敗しました。データ60枚の再現と既存の生成 fixture でのモデル一致は成功しましたが、bytes が異なる原因は未確定です。plan/hash/許容差を変えず、通常実行と `--check` の strict 条件を保持します。CI は別の非測定モード `--check-app` を使い、artifact 自身の manifest/hash・同じ checkpoint/revision・dist 整合と App/helper 起動だけを確認します。実測 artifact と異なる場合は `measurementModelMatches=false` を出し、科学的比較の再現とは扱いません。最新コミットの CI 判定とマージ状況は [PR #7](https://github.com/ttomohisa/open-vision-builder/pull/7) を参照してください。

canonical PNG・花5クラス・Small の固定 backbone とヘッド学習・当該 Linux/Chromium WASM に限る部分結果です。一般の JPEG decode、WebGPU、検出、G4 全体・P1 全体・製品リリースを認定しません。次は分類のローカル保存・復元、資源量、中断・再開へ進みます。

### PR #6: 公開実画像での限定比較を完了、条件未達を保存

[事前登録した計画](superpowers/plans/2026-10-25-dfine-own-prefix-impact.md)で、[公開実画像](verification/2026-10-25-impact-data.md)の4 train / 2 tuning / 4 evaluation、全58注釈、同一の12 SGD step を比較しました。[実測と次の判断](verification/2026-10-25-own-prefix-impact.md)・[比較 JSON](verification/evidence/2026-10-25-own-prefix-impact.json)を保存しています。整数 timeout の初回実行不具合は資料を残して修正し、データ・計画を変えず一回の完全実行を行いました。

0/6/12 step の同一特徴対照、全12 step の matching・最終 logits/boxes・損失・全勾配・更新値の部品比較は通りました。中間 `corners` は9/12 stepで不一致です。平均学習損失の改善は browser **0.004070%** / PyTorch **0.003928%** で、事前条件の1%に届きません。評価4画像の AP50 は両実装・更新前後で0.8463458012でしたが、有効な追加学習や実用品質を認定する結果ではありません。

完全 ONNX は fresh browser context で10/10画像が一致し、固定 initializer 576個と graph が不変、更新8テンソルの格納も一致しました。**Python CPU では6012/12667の2画像が元の最終出力条件外**です。6012は box の最大絶対差が0.3136631250あり、小さな丸め差と断定しません。学習効果・export の両方を条件未達として、追加測定をここで止めました。

評価画像は今回の追加学習に未使用という意味で、元 COCO checkpoint のモデル選定との独立は保証しません。画像単位の小規模比較であり、G4 全体や実用精度を認定しません。CI はデータ再現・単体テスト・ブラウザ起動の事前確認を行い、毎回の追加学習実験は自動実行しません。

| 項目 | 確認できた範囲 / 残課題 |
| --- | --- |
| 分類 UI | クラス定義、画像取込、明示的な学習/評価分割、実行、中断、評価、ZIP。矩形注釈・永続保存は未実装 |
| 分類モデル | MobileNetV4 Conv Small の正確な safetensors revision・ハッシュを固定。完全な公開モデルを準備・元 PyTorch と照合 |
| ブラウザ学習 | jax-js で新しい分類ヘッドを実更新。花写真30枚・40 updatesで train 30/30、独立 PyTorch と同条件比較。特徴抽出部分は固定、WASM |
| 完全 ONNX | 固定重みと更新したヘッドを内包する単一 FP32 ONNX。花写真50枚で fresh browser と Python が条件内。元 controller の通信検査 ERROR は別に保持 |
| Python 利用 | ブラウザ生成モデルを ONNX Runtime CPU で実行。花写真50枚の logits を追加変換なしで比較し条件内 |
| GridSample | 固定した ESM/CJS の4行を hash 検証付きで修正。6手計算・12独立 PyTorch fixture の値・特徴/座標勾配を Node / browser WASM で比較 |
| 検出の限定学習 | D-FINE Nano の最終 pretrained bbox / class、計60,372パラメータ。元の2 step の部品検証に加え、実画像の各 runtime 自身の特徴で12 step を比較。両実装の損失改善は約0.004%で、1%基準に未達 |
| 検出の完全 ONNX | 更新した8テンソルを元の全グラフへ反映。従来の合成 fixture は照合済み。新しい実画像では fresh browser 10/10一致、Python CPU 8/10一致・2/10不一致。完全モデル生成と runtime 一致の未達を区別 |
| 検出の未解決 | **G1: 中間特徴量の一致は FAIL**。全3入力の元 PyTorch / ORT 比較を保存。最初の Conv の FP32 差と後段での増幅を確認したが、全演算の同等性を証明したものではない。検出 UI・独自クラス・完全 fine-tuning・精度は未実装/未検証 |
| 実データ品質 | 分類の花5クラスは browser/native とも評価15/20、macro F1 0.7504761905で限定条件内。controller ERROR は保持し、G4全体・一般化品質を認定しない。検出は事前の学習改善条件未達。Mediumは未検証 |
| 環境 | Linux x86_64 / Headless Chrome 156.0.8078.4 / WASM、非隔離サブパス。一般 PC、Edge、WebGPU、edge 端末は未認定 |
| 資源と継続性 | 経路の経過時間をレポート。ピークメモリ未測定。中断・古い結果の無効化は検証済み。保存・再開・バックアップは未実装 |
| データ非送信 | 既存の公開生成データ試験に加え、花写真試験でも記録した要求は同一 origin の GET。strict controller は任意 build-info GET 一件で ERROR を保持。運営 API / DB なし |
| GitHub Pages | PR #3 の実サイトで分類学習・ZIP・Python 一致を検証済み。新しい公開は commit と Actions を再確認。検出の未認定モデルは Pages に含めない |
| Cloudflare | 静的 dist と base 指定手順を用意。Cloudflare への実デプロイは未実施 |
| 利用条件 | 分類 Small と検証用 D-FINE Nano COCO の正確な重み・変換元・Apache 表示を固定し、成果物に同梱。Objects365 重みは使わない。アプリ本体のライセンスは未選定 |

## 証拠

- [ブラウザ分類・Python 一致の記録](verification/2026-10-24-browser-classification.md): 実行コマンド、環境、測定結果、制約。
- [分類の実画像比較結果](verification/2026-10-25-classification-real-images.md)・[比較証拠](verification/evidence/2026-10-25-classification-real-images.json): 一回の科学的比較 PASS、誤分類5枚、独立 export の各50件一致と、元 controller の ERROR。
- [分類の固定実画像と出典](verification/2026-10-25-classification-data.md)・[実画像比較の実装計画](superpowers/plans/2026-10-25-classification-real-images.md): 事前登録したデータ・条件と、結果を変更しない打切り記録。
- [公開モデル準備](verification/2026-10-24-model-preparation.md): revision、SHA-256、完全 ONNX、PyTorch と ORT の比較。
- [D-FINE に関係する演算 probe](verification/2026-10-24-dfine-probe.md): 固定した実装、再現値、失敗条件。
- [GridSample の修正](verification/2026-10-24-grid-sample-correction.md): 元の失敗と、固定 hash の修正後の独立比較。
- [D-FINE 基礎検証](verification/2026-10-24-dfine-foundation.md): 最終ヘッド・完全 ONNX の部分成功と、変更していない中間特徴ゲートの失敗。
- [D-FINE 実画像の学習・出力影響](verification/2026-10-25-own-prefix-impact.md): 同一特徴対照の成功、実経路の比較、学習効果と Python export の条件未達、打切り判断。
- [ADR 0002](decisions/0002-classification-spike.md): 分類の部分実装を先に確認し、検出ゲートを継続する理由。
- [分類の実装計画](superpowers/plans/2026-10-24-classification-spike.md): PR #3 の作業範囲。
- [ADR 0003](decisions/0003-dfine-foundation.md) と [検出の実装計画](superpowers/plans/2026-10-24-dfine-foundation.md): 部分検証と未解決条件を分けて進める判断。

学習対象がヘッドだけであること、評価データの種類と固定分割、資源量・一般的な実用精度が未確認であることを UI とレポートにも表示します。上流の公開精度を本実装の達成精度として使いません。

## 次の作業

1. **PR #7 に科学的比較の結果と元 controller の ERROR を保存する。** 通信検査の扱いはモデル推論を行わない事前確認で修正済み。別の CI 起動確認と結果レビューを確認する。今回の学習・モデル推論は一回で終了する。
2. **分類プロジェクトのローカル保存・復元とバックアップを次の実装単位にする。** 画像・ラベル・分割・学習条件・学習済み状態を失わない範囲を先に定め、既存の ONNX 持ち出しと分けて検証する。
3. 分類の対象 PC / WASM で時間・ピークメモリ・操作応答・中断と再開を確認する。WebGPU、Medium、検出は今回の成功から対応済みにしない。[ROADMAP](ROADMAP.md) の P1 の資源・継続性と P2 の保存・復元を段階的に進め、リリース判定は [VERIFICATION](VERIFICATION.md) の全ゲートに従う。

検出は別の保留課題です。PyTorch も1%の学習改善条件に届かなかった限定レシピ・学習量・データ適合性、および6012/12667の Python export 不一致を残します。元 G1 FAIL、検出 UI 非公開を維持し、新たな測定には [ADR 0005](decisions/0005-bound-numerical-investigation.md) の問い・修正仮説・結果別判断・予算を必要とします。分類の次作業として detector replay や許容差調整を自動追加しません。

ONNX 必須、ブラウザ学習、商用利用可能な重み、実際の検出器学習という要件は変更していません。
