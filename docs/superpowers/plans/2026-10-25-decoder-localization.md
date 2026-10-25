# 矩形入力の decoder 数値差を絞る実装計画

> **For agentic workers:** Use superpowers:executing-plans and systematic-debugging. Check off actual execution, not intended results.

**Goal:** 保存済みの矩形1画像で、中間特徴の差が増える decoder 段と、同一入力でも差が出る部品を区別する。

**Architecture:** 既存の矩形診断に、固定した元 PyTorch module の hook と ONNX 中間出力の観測を追加する。各 decoder layer と bbox MLP に ONNX 側の同一入力を渡して再計算する。元の prefix 出力が bit 単位で変わらないことを必須とする。

**Tech Stack:** 既存の固定 PyTorch / Transformers / ONNX / ONNX Runtime CPU。依存追加なし。

**Spec:** [ADR 0004](../../decisions/0004-dfine-investigation-checkpoint.md)、[元の最小再現](../../verification/2026-10-25-rectangle-reproduction.md)、[STATUS](../../STATUS.md) の次の作業1。

## 制約と判断

- 起点は PR #5 の `f1aa65498d6e381ef31b972cc0275a40acec618d`。同じ専用ブランチで続け、main へ変更せず、マージはユーザーが行う。
- ユーザーの「引き続き」「マージが必要なときは教えて」に基づき、既存設計内の原因調査を進める。製品要件変更の承認は含まない。
- 固定した COCO-only Nano、入力 hash、元の6出力、FP32、CPU1スレッド、ORT_ENABLE_ALL を維持する。
- feature は元の atol=2e-4 / rtol=1e-4、reference 座標は atol=2e-5 / rtol=1e-4。query 順序の入替や最大観測値に合わせた閾値変更をしない。
- 新しい診断結果でも `qualifiesDetector=false`。終了0は測定完了のみ。元の G1、製品 UI、学習範囲、品質の判定は変えない。
- ソースを追跡すると、既存の `references` は layer 0 後に作る固定 anchor で、後段へ渡す refined reference とは別。各段で消費される座標も観測する。
- コードと小さい JSON を節目で GitHub へ保存する。巨大 tensor、モデル、ユーザーデータはコミットしない。

## Review Focus

- 同名 node、誤った operator、欠落した tap を妥当そうな比較結果として採用しない。
- 同じ shape でも違う意味の anchor / refined reference を取り違えない。
- 中間出力追加で最適化が変わった場合、元の計算の診断と主張しない。
- replay に元 PyTorch の入力が混入しない。hook は replay 前と例外時に外す。
- 数値 FAIL を完了扱いの終了コードや CI 成功で隠さず、型・shape・非有限値の誤りは停止する。

## Task 1: decoder の境界と同一入力 replay

**Files:** `scripts/diagnose_detector_decoder.py`（新規）、`scripts/reproduce_detector_rectangle.py`、`tests/detector_decoder_test.py`（新規）。

**Interfaces:** 元の診断は source / graph hash と6出力を検証する。新しい helper は元 module の捕捉、固定 graph の tap 対応、同一 ORT 入力での再計算結果を返す。report に `decoder` を追加する。

- [x] 専用ブランチと GitHub head の一致を確認。既存 Python 47件成功。変更前の矩形診断を再実行。
- [x] 座標の元の許容差が適用されるテストの RED を確認し、shape / 非有限値・hook の所有コピーと解除・全 replay 入力・空間 shape・query 順序の6テストを実装した。graph 境界は固定 hash / operator 検査と独立した source / graph 対応レビューで確認した。
- [x] 3 decoder layer、pre_bbox、bbox 0/1 の実入出力と消費 reference を捕捉する。全 layer の hidden / position / reference / memory を ORT から与えて元 module を再実行する。
- [x] 元 prefix 4出力の bit 一致を確認し、元の数値条件で測定する。数値結果を未解決事項とともに保存する。
- [x] Python 全53テストと実モデル診断を実行し、コード・測定 JSON を今回のコミットに含める。GitHub 保存は最後に SHA で確認する。

## Task 2: CI、記録、レビュー

**Files:** `.github/workflows/ci.yml`、`docs/verification/2026-10-25-decoder-localization.md`、`docs/verification/evidence/2026-10-25-decoder-cpu.json`、`docs/STATUS.md`、関連する入口文書。

**Interfaces:** Task 1 の decoder report を既存の CI summary に掲載する。数値判定と測定完了を区別する。

- [x] CI が新しい観測を実行し、不完全な結果を拒否して、差が出た段を summary に表示する。抽出した summary 処理を今回の実測 JSON で実行し、FAIL / PASS の表を確認した。
- [x] 実測、再実行コマンド、残る不確実性、次の調査点を保存する。
- [x] 独立コード・CI・記録レビューで Critical / Important / Minor 指摘なし。
- [ ] 最新 head の GitHub CI を確認する。最終結果は PR のチェックと説明に記録する。
- [ ] PR #5 の説明と GitHub 上の保存 SHA を確認。マージを要する区切りならユーザーに伝える。

## 実行台帳

- Pre-flight: Task 1 の report を Task 2 の CI が読む。既存 `comparisons` / `sourceParityPassed` は維持し、decoder の数値結果だけで元の資格判定を変更しない。
- Ruling: 専用ブランチの既存 checkout を継続使用する。ユーザーが同じ PR での継続を指示し、作業ツリーは clean、他の担当は read-only のため追加 worktree は不要。
- Task 1: complete — Python 53件成功、実モデル測定 completed / sourceParityPassed=false。局所6部品の replay は許容差内、元 prefix 4出力は instrumentation 前後で bit 一致。
- Task 2: local checks and independent review complete — workflow YAML と実際の summary 処理を確認。GitHub へ保存し、最新 CI の完了を確認する。
