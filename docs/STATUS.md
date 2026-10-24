# 現在の状態

更新日: 2026-10-24 JST。

## 段階

**P1a: 分類のブラウザ学習と完全 ONNX 出力を実装・ローカル検証済み。P1 全体と初期版は未完了です。**

合意・検証方針は [PR #1](https://github.com/ttomohisa/open-vision-builder/pull/1) と [PR #2](https://github.com/ttomohisa/open-vision-builder/pull/2) で記録しました。本実装の比較元は `64e9fe53579ca95d464ed85944f9bd3ef237336e`、ブランチは `feat/browser-classification-spike` です。実装のマージ状況は [PR 一覧](https://github.com/ttomohisa/open-vision-builder/pulls)を確認してください。

## 実装・検証の状態

| 項目 | 確認できた範囲 / 残課題 |
| --- | --- |
| 分類 UI | クラス定義、画像取込、明示的な学習/評価分割、実行、中断、評価、ZIP。矩形注釈・永続保存は未実装 |
| 分類モデル | MobileNetV4 Conv Small の正確な safetensors revision・ハッシュを固定。完全な公開モデルを準備・元 PyTorch と照合 |
| ブラウザ学習 | jax-js で新しい分類ヘッドを実更新。特徴抽出部分は固定。WASM 単一スレッドで検証 |
| 完全 ONNX | 固定重みと更新したヘッドを内包する単一 FP32 ONNX。別のブラウザ実行セッションで一致確認 |
| Python 利用 | ブラウザ生成モデルを ONNX Runtime CPU で実行し、公表 fixture の logits を比較。追加変換なし |
| 物体検出 | **BLOCKED**。jax-js ONNX GridSample の端付近の forward / gradient 不一致を独立 probe で再現。D-FINE 全体の学習は未実施 |
| 実データ品質 | 未検証。生成サンプルは経路確認用。Medium、独自タスク、同一条件 PyTorch 学習比較は未実施 |
| 環境 | Linux x86_64 / Headless Chrome 156.0.8078.4 / WASM、非隔離サブパス。一般 PC、Edge、WebGPU、edge 端末は未認定 |
| 資源と継続性 | 経路の経過時間をレポート。ピークメモリ未測定。中断・古い結果の無効化は検証済み。保存・再開・バックアップは未実装 |
| データ非送信 | 公開生成データのブラウザ試験で外部通信・GET 以外の通信なし。運営 API / DB なし |
| GitHub Pages | workflow と初回設定手順を実装。実際の公開はリポジトリ設定・Actions の結果で確認する |
| Cloudflare | 静的 dist と base 指定手順を用意。Cloudflare への実デプロイは未実施 |
| 利用条件 | 固定した Small のモデル出典と Apache 表示を同梱。依存物の表示を静的配布に含める。アプリ本体のライセンスは未選定 |

## 証拠

- [ブラウザ分類・Python 一致の記録](verification/2026-10-24-browser-classification.md): 実行コマンド、環境、測定結果、制約。
- [公開モデル準備](verification/2026-10-24-model-preparation.md): revision、SHA-256、完全 ONNX、PyTorch と ORT の比較。
- [D-FINE に関係する演算 probe](verification/2026-10-24-dfine-probe.md): 固定した実装、再現値、失敗条件。
- [ADR 0002](decisions/0002-classification-spike.md): 分類の部分実装を先に確認し、検出ゲートを継続する理由。
- [実装計画](superpowers/plans/2026-10-24-classification-spike.md): この PR の作業範囲。

学習対象がヘッドだけであること、サンプルが合成データであること、資源量・実用精度が未確認であることを UI とレポートにも表示します。上流の公開精度を本実装の達成精度として使いません。

## 次の作業

1. **P1b: 検出の障害を解消する。** GridSample の forward・特徴側/座標側 gradient を元実装と確認し、修正した jax-js または別の明示的な学習実装を選ぶ。無検証の依存差し替えで済ませない。
2. 固定した COCO 学習済み D-FINE の完全な training forward・損失・matching・クラス変更を移植し、事前学習済み位置推定重みの更新を確認する。
3. 分類・検出をそれぞれ実データの未使用評価分割と、同条件の PyTorch 学習で比較する。分類バックボーンの追加学習、Medium / Small の品質候補を評価する。
4. WebGPU と対象 PC の時間・ピークメモリ、中断・checkpoint 再開を検証し、P1 全体の採否を判断する。
5. [ROADMAP](ROADMAP.md) の P2 でプロジェクト保存・バックアップと注釈 UI を整える。リリース判定は [VERIFICATION](VERIFICATION.md) の全ゲートに従う。

ONNX 必須、ブラウザ学習、商用利用可能な重み、実際の検出器学習という要件は変更していません。
