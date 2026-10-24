# D-FINE 学習経路の基礎検証

日付: 2026-10-24 JST。比較元: PR #3 をマージした `037ef78e6e46c5191a11c7556c3127f2b28d6f15`。

## 目的と承認済みの範囲

ユーザーの「マージした。次に進んで」に従い、[STATUS](../../STATUS.md) と [ADR 0002](../../decisions/0002-classification-spike.md) の P1b を進める。ブラウザ内の実学習、完全 ONNX、API / DB 不要、商用利用条件という製品要件は維持する。下記の実装上の選択は実装者の判断として記録する。

次のレビュー単位は、GridSample の修正と、COCO 学習済み D-FINE Nano の限定した学習・書き出し経路を再現可能にすること。元の detector 全体の学習、独自クラス、実用精度、注釈 UI、WebGPU の完成とは区別する。

## 選択肢と判断

| 選択肢 | 判断 |
| --- | --- |
| jax-js ONNX の検証済み配布物に最小の修正を明示的に適用 | 採用。公開 API に演算差し替え口がないため、版・元の全ファイル hash・修正後 hash を確認して ESM/CJS を揃える。未知の版や差分はエラーにする |
| ONNX importer 全体を fork / vendor | 今回は採らない。必要な変更は GridSample の隣接 index を求める順序で、loader 全体の独自保守は不要 |
| TFJS へ学習エンジンを変更 | 比較候補として維持。単一演算の再現可能な不具合に対し、モデルと全損失を別エンジンへ移植する判断はまだしない |

## 1. GridSample

`@jax-js/onnx@0.1.2`、`@jax-js/jax@0.1.27` を維持する。左・上の隣接 index は clip 済みの floor から作らず、元の floor と floor+1 をそれぞれ clip する。既存の四近傍の範囲判定と tensor の参照所有権を維持する。

6点の既存手計算 fixture は絶対誤差 `1e-5` のまま使う。追加の独立 PyTorch fixture は、forward `atol=1e-5, rtol=1e-5`、feature/grid VJP `atol=2e-5, rtol=1e-5` を実行前に固定する。非正方形、複数 batch/channel/output、画像端・外側、align_corners の両方、縦横が1の入力を含める。

## 2. 正確な事前学習済み detector

候補の `ustc-community/dfine-nano-coco`、revision `066438d3d8f0da137a37b38fdf3368fd4afceced` を使用する。この revision が Nano attention の変換修正を含むことを履歴と重みで確認する。`model.safetensors` は SHA-256 `19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b`、15,278,996 bytes。検証してから読み込み、Objects365 の重みへ置き換えない。

Transformers 4.56.0 の固定実装と元 D-FINE の来歴・Apache 表示を記録する。HF 実装との数値比較を、未実施のネイティブ `.pth` 比較と呼ばない。640×640 / batch 1 / 80 COCO classes / 300 queries の FP32 完全モデル、元の logits と正規化 cxcywh boxes を扱う。検証用の前処理は固定し、上流 AP を自分たちの精度として使わない。

準備時の PyTorch / ORT 比較は logits `atol=2e-4, rtol=1e-4`、boxes `atol=2e-5, rtol=1e-4` を事前に固定する。公開の手続き的 fixture を使い、全出力の shape、有限値、各要素を確認する。operator / domain / opset / dtype / external data を一覧化する。

## 3. 限定した実学習

最終 decoder の事前学習済み bbox MLP と class projection を更新し、それ以前の detector と LQE は固定する。学習の入力は、実際の frozen prefix が計算した最終 hidden state、bbox 入力、前段の corner residual、基準 boxes。位置分布の softmax / integral / distance-to-box / LQE を元実装に合わせる。

最初の成立確認では既存80クラスを維持する。独自クラスへの対応には encoder、全 decoder、必要な埋め込みの変更も必要であり、最終射影の形だけを変えて完成としない。

Hungarian matching は微分しない。最終段の VFL / L1 / GIoU と SGD を独立 PyTorch 実装と比較し、事前学習済み bbox 重みの勾配・実更新・次回 forward の変化を確認する。元レシピの FGL / DDF、補助出力、denoising、全層の matching 統合を含む学習と同一とは扱わない。採用する限定レシピと勾配許容差は実装・実測記録へ固定する。

## 4. 完全 ONNX と検証

元の完全 inference graph に対し、対応表で指定した initializer だけを最終状態へ置換する。shape、dtype、名前、共有 alias、有限値を検査し、固定パラメータを欠落させない。学習専用の追加出力を取り除き、単一ファイルだけで別の ORT Web session / Python CPU が推論できることを確認する。

全重みや生成 fixture の巨大 tensor は ignored の artifact に保存する。必要な公開モデルとソース表示だけを静的配布し、1 asset 25 MiB を確認する。ユーザーデータを CI へ送る機能は作らない。

## 成立条件と未完了の扱い

修正前の不一致、修正後の値と両勾配、完全モデルの出典と推論、限定学習の元実装比較、完全 ONNX の独立検証を別々に記録する。失敗した場合は原因と残る条件を記録し、成功した項目で他のゲートを置き換えない。製品全体の G0–G9 は [VERIFICATION](../../VERIFICATION.md) を維持する。
