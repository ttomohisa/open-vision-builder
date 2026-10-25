# 現在の状態

更新日: 2026-10-25 JST。

## 段階

**P1a の分類はマージ済み。P1b は検出の最終ヘッド学習・完全 ONNX の基礎検証まで進行。検出の中間特徴量一致は FAIL、P1 全体と初期版は未完了です。**

合意・検証方針は [PR #1](https://github.com/ttomohisa/open-vision-builder/pull/1) と [PR #2](https://github.com/ttomohisa/open-vision-builder/pull/2) で記録しました。[分類の PR #3](https://github.com/ttomohisa/open-vision-builder/pull/3)、[検出基礎検証の PR #4](https://github.com/ttomohisa/open-vision-builder/pull/4)、[再現・局所診断の PR #5](https://github.com/ttomohisa/open-vision-builder/pull/5) はマージ済みです。PR #5 のマージは 2026-10-25 00:13 UTC（09:13 JST）、比較元は `f73cdb9407f60fb402e23afa9abdefdd7e98e058`、今回のブランチは `test/dfine-own-prefix-impact` です。以後のマージ状況は [PR 一覧](https://github.com/ttomohisa/open-vision-builder/pulls)を確認します。

## PR #4 後の作業と復元範囲

追加6入力で元実装と各 runtime 自身の特徴を使う学習を比較した際、候補契約は全件不合格でした。矩形入力では最終出力も不一致、ブラウザでは近い score の query 順序の入替を観測しました。学習の損失・勾配・更新値に部分一致の記録がありますが、検出器の合格にはしていません。

**追加の未 push コード・生レポート・ローカルコミットは、作業環境の自動整理で失われました。GitHub 上の PR #4 までの履歴は維持されています。** 現在の追加成果物は、会話の実行記録から復元した[調査メモ](verification/2026-10-24-dfine-prefix-investigation.md)と[再開方針](decisions/0004-dfine-investigation-checkpoint.md)です。追加の検証コードや、当時記録したテスト成功を、このブランチで実行・保存済みのものとして数えません。

PR #5 の初回コミットは文書のみでした。その後、[矩形1画像の最小再現](verification/2026-10-25-rectangle-reproduction.md)を再構築し、診断コードと小さい測定 JSON を保存しました。今回は最終出力は許容差内、中間3出力は不一致でした。過去メモの最終出力不一致を完全に再現したものではありません。下表の学習・export は PR #4 までの範囲です。

続く [decoder の切り分け](verification/2026-10-25-decoder-localization.md)では、最初の corner MLP が小さな入力差を増幅することを確認しました。同一 ONNX 入力を元 module に渡すと3 MLP は完全一致、3 decoder layer も許容差内でした。各 runtime 自身の入力では中間特徴の不一致が残り、G1 は FAIL のままです。

ユーザーの指示に基づく[方針点検](decisions/0005-bound-numerical-investigation.md)で、数値調査の自動延長を止めました。PR #5 は現在の再現・局所診断で区切り、予定していた decoder 全体 replay は保留します。次は学習・出力への影響を優先し、分類の実データ品質確認も独立して進める方針です。今回の点検で新しい学習・精度実験を実行したものではありません。

## 実装・検証の状態

### 次の限定比較の事前登録

[実行計画](superpowers/plans/2026-10-25-dfine-own-prefix-impact.md)と[公開実画像の記録](verification/2026-10-25-impact-data.md)を用意しました。4 train / 2 tuning / 4 evaluation、全58注釈、同一の12 SGD step、0/6/12 step の同一特徴対照、fresh browser context と Python の完全 ONNX 検証です。データ manifest と計画の hash を結び付け、条件を変えた開始を拒否します。この段階では新しい学習実験は NOT RUN です。

評価画像は今回の追加学習に未使用という意味で、元 COCO checkpoint のモデル選定との独立は保証しません。画像単位の小規模比較であり、G4 全体や実用精度を認定しません。CI はデータ再現・単体テスト・ブラウザ起動の事前確認を行い、毎回の追加学習実験は自動実行しません。

| 項目 | 確認できた範囲 / 残課題 |
| --- | --- |
| 分類 UI | クラス定義、画像取込、明示的な学習/評価分割、実行、中断、評価、ZIP。矩形注釈・永続保存は未実装 |
| 分類モデル | MobileNetV4 Conv Small の正確な safetensors revision・ハッシュを固定。完全な公開モデルを準備・元 PyTorch と照合 |
| ブラウザ学習 | jax-js で新しい分類ヘッドを実更新。特徴抽出部分は固定。WASM 単一スレッドで検証 |
| 完全 ONNX | 固定重みと更新したヘッドを内包する単一 FP32 ONNX。別のブラウザ実行セッションで一致確認 |
| Python 利用 | ブラウザ生成モデルを ONNX Runtime CPU で実行し、公表 fixture の logits を比較。追加変換なし |
| GridSample | 固定した ESM/CJS の4行を hash 検証付きで修正。6手計算・12独立 PyTorch fixture の値・特徴/座標勾配を Node / browser WASM で比較 |
| 検出の限定学習 | D-FINE Nano の最終 pretrained bbox / class、計60,372パラメータを更新。80 COCO クラス・前段・LQE は固定。同一特徴の独立 PyTorch 比較と browser で2 SGD step を検証 |
| 検出の完全 ONNX | 更新した8テンソルを元の全推論グラフへ反映。別 ORT Web session / Python CPU で24,000 logits・1,200 boxes を照合。検証用 ZIP に利用例・出典・FAIL 状態を同梱 |
| 検出の未解決 | **G1: 中間特徴量の一致は FAIL**。全3入力の元 PyTorch / ORT 比較を保存。最初の Conv の FP32 差と後段での増幅を確認したが、全演算の同等性を証明したものではない。検出 UI・独自クラス・完全 fine-tuning・精度は未実装/未検証 |
| 実データ品質 | 未検証。生成サンプルは経路確認用。Medium、独自タスク、同一条件 PyTorch 学習比較は未実施 |
| 環境 | Linux x86_64 / Headless Chrome 156.0.8078.4 / WASM、非隔離サブパス。一般 PC、Edge、WebGPU、edge 端末は未認定 |
| 資源と継続性 | 経路の経過時間をレポート。ピークメモリ未測定。中断・古い結果の無効化は検証済み。保存・再開・バックアップは未実装 |
| データ非送信 | 公開生成データのブラウザ試験で外部通信・GET 以外の通信なし。運営 API / DB なし |
| GitHub Pages | PR #3 の実サイトで分類学習・ZIP・Python 一致を検証済み。新しい公開は commit と Actions を再確認。検出の未認定モデルは Pages に含めない |
| Cloudflare | 静的 dist と base 指定手順を用意。Cloudflare への実デプロイは未実施 |
| 利用条件 | 分類 Small と検証用 D-FINE Nano COCO の正確な重み・変換元・Apache 表示を固定し、成果物に同梱。Objects365 重みは使わない。アプリ本体のライセンスは未選定 |

## 証拠

- [ブラウザ分類・Python 一致の記録](verification/2026-10-24-browser-classification.md): 実行コマンド、環境、測定結果、制約。
- [公開モデル準備](verification/2026-10-24-model-preparation.md): revision、SHA-256、完全 ONNX、PyTorch と ORT の比較。
- [D-FINE に関係する演算 probe](verification/2026-10-24-dfine-probe.md): 固定した実装、再現値、失敗条件。
- [GridSample の修正](verification/2026-10-24-grid-sample-correction.md): 元の失敗と、固定 hash の修正後の独立比較。
- [D-FINE 基礎検証](verification/2026-10-24-dfine-foundation.md): 最終ヘッド・完全 ONNX の部分成功と、変更していない中間特徴ゲートの失敗。
- [ADR 0002](decisions/0002-classification-spike.md): 分類の部分実装を先に確認し、検出ゲートを継続する理由。
- [分類の実装計画](superpowers/plans/2026-10-24-classification-spike.md): PR #3 の作業範囲。
- [ADR 0003](decisions/0003-dfine-foundation.md) と [検出の実装計画](superpowers/plans/2026-10-24-dfine-foundation.md): 部分検証と未解決条件を分けて進める判断。

学習対象がヘッドだけであること、サンプルが合成データであること、資源量・実用精度が未確認であることを UI とレポートにも表示します。上流の公開精度を本実装の達成精度として使いません。

## 次の作業

1. **各実行系自身の特徴を使う学習・出力への影響を測る計画を先に固定する。** [ADR 0005](decisions/0005-bound-numerical-investigation.md) の結果別判断に従い、公開注釈データ・分割・同じ限定学習レシピ・数値条件・実行予算を定め、実ブラウザと PyTorch を比較する。同一特徴の対照、matching、全勾配・更新、未使用画像の結果、完全 ONNX を含める。元の G1 は FAIL のまま維持し、検出 UI はまだ接続しない。decoder 全体 replay の自動追加はしない。PR #5 の確認後にマージを案内し、実装は別 PR へ分ける。
2. 限定した最終段から完全な training forward・補助/分布損失・matching・独自クラスへ拡張する。encoder と全 decoder のクラス依存部分を扱い、位置推定を無作為に初期化し直さない。現在の2 step を過適合・実用精度の証拠にしない。
3. 分類・検出をそれぞれ実データの未使用評価分割と、同条件の PyTorch 学習で比較する。分類バックボーンの追加学習、Medium / Small の品質候補を評価する。
4. WebGPU と対象 PC の時間・ピークメモリ、中断・checkpoint 再開を検証し、P1 全体の採否を判断する。
5. [ROADMAP](ROADMAP.md) の P2 でプロジェクト保存・バックアップと注釈 UI を整える。リリース判定は [VERIFICATION](VERIFICATION.md) の全ゲートに従う。

ONNX 必須、ブラウザ学習、商用利用可能な重み、実際の検出器学習という要件は変更していません。
