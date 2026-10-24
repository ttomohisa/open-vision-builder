# ADR 0003: 検出の最終ヘッドと完全 ONNX を検証し、中間特徴の失敗を独立して残す

- 日付: 2026-10-24 JST。
- 状態: P1b の基礎実装に限定して採用。検出器の製品採用と G1 は保留。
- 比較元: PR #3 マージ後の `037ef78e6e46c5191a11c7556c3127f2b28d6f15`。
- 必須要件: [PRODUCT](../PRODUCT.md)、[VERIFICATION](../VERIFICATION.md)。変更なし。

## 背景

ユーザーの「マージした。次に進んで」に従い、分類に続いて検出の実学習と完全 ONNX を検証した。元の GridSample は左・上の隣接 index の作り方に誤りがあり、固定版の4行修正で独立した forward / 両 VJP の比較が通った。これだけで D-FINE の training graph 全体が成立するとは判断しない。

D-FINE の deploy graph は native training forward の detach、補助出力、denoising、FGL/DDF、全層 matching をそのまま表現していない。現在の jax ONNX importer には BatchNormalization、Range、ScatterND 等の未登録演算もある。完全グラフを読み込んで入力を微分する操作を、事前学習済み検出器の学習と呼ぶことはできない。

## 今回の学習範囲

固定した COCO-only Nano の safetensors を起点とする。元の80クラスを維持し、最終 decoder の pretrained bbox MLP と class projection の8テンソル、60,372パラメータを学習する。前段の特徴抽出・decoder と最終 LQE は固定する。学習済み位置推定を無作為に初期化し直さない。

損失は最終段の VFL + 5 L1 + 2 GIoU、optimizer は FP32 の明示的 SGD とする。Hungarian matching、IoU の教師値、VFL の教師重みは autodiff の外で計算する。同じ元特徴・初期重み・ラベル・step 数を与えた独立 PyTorch 実装と、全パラメータ勾配・更新後の値を比較する。

完全 ONNX exporter は実テンソルで検証した source-key / initializer / shape 対応を使い、全固定グラフと学習後の8テンソルを1ファイルへまとめる。ブラウザの新しいセッションと Python CPU で生出力を比較し、kit は追加モデル取得・PyTorch・変換処理なしで実行できる。

## 残った失敗と判断

元 PyTorch と ONNX の3入力比較では、最終 logits・boxes と reference boxes は事前定義の許容差を満たす。一方、bboxInput / classInput / residualCorners に不合格が残る。ブラウザでも phase 0 の residualCorners に不合格がある。**これを G1 FAIL として保持し、最終出力やヘッド学習の成功で置き換えない。**

最初の Conv から小さな FP32 差があり、調べた27項の内積では PyTorch / ORT の双方が FP64 参照に対する丸め誤差の上界内だった。同一 ORT 入力を渡した以前の bbox MLP は PyTorch と完全一致し、入力差が後段で増幅することと整合する。ただし、この観測は前段の全演算の同等性を証明しない。

ORT 最適化設定・PyTorch MKLDNN 設定・固定 BN の affine 事前計算では元ゲートを解消しなかった。FP64 prefix も元 FP32 参照とのゲートを満たさず、約30.5 MB、ORT CPU の DOUBLE Conv 非対応という制約があるため採用しない。許容差も変更しない。

| 選択肢 | 判断 |
| --- | --- |
| 観測した最大誤差より大きく許容差を変える | 採らない。元の失敗を残す |
| 最終出力の合格だけで detector 全体を認定する | 採らない。G1 は特徴段も要求する |
| 全体を別エンジン/精度へ即時移植する | 保留。今回得た原因・部品検証を失わず、次の比較仕様を先に決める |
| 部品成功と未解決の全体ゲートを明示した基礎 PR | 採用。ROADMAP が認める途中結果のレビュー単位とし、製品の検出機能へは接続しない |

## CI と公開の境界

`prepare_detector.py` は全3入力6出力を評価し、不合格なら非0終了する。調査用ファイルと `passed:false` の manifest/report を専用の ignored ディレクトリに残し、合格モデルの出力先には公開しない。source/hash/構造の失敗は通常の数値差と混同しない。

CI はこの厳格な結果を summary に明示し、完全な失敗レポート・source・最終出力を検査してから部品検証を継続する。missing/invalid な証拠や head/export の失敗は CI を止める。ブラウザと kit の `detectorQualification` に失敗を引き継ぐ。検出の調査用モデルは `public/` / `dist/` へ入れず、Pages と製品のモデル選択から除外する。

このため、基礎 PR の回帰テストが成功しても、G1 や P1b 全体の合格を意味しない。[実測記録](../verification/2026-10-24-dfine-foundation.md)と [STATUS](../STATUS.md) を正本にする。

## 次の判断

1. 前段の誤差伝播を調べ、必要な場合は共有の数値誤差予算と新しい未使用入力の検証手順を事前に定める。過去の失敗は削除しない。
2. 独自クラスの encoder / 全 decoder / embedding を扱い、補助・分布損失を含む学習レシピへ拡張する。
3. 少量過適合と未使用実データの品質を、同一条件の PyTorch 学習と比較する。今回の2 SGD step はその代わりにならない。
4. 実機・WebGPU・ピークメモリ・中断/保存/再開を測り、採用範囲を判断する。
