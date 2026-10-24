# 矩形1画像の最小再現

実行日: 2026-10-25 JST。起点: PR #5 の `321b3a696ca68ff25d2c1680e1576759f1921363`。今回再構築した[診断コード](../../scripts/reproduce_detector_rectangle.py)と[測定 JSON](evidence/2026-10-25-rectangle-cpu.json)を保存する。

**今回の CPU 実行では最終 logits / boxes / references は元の許容差内だった。bboxInput / classInput / residualCorners の不一致は再現した。** 元の G1 は FAIL のまま。過去の[復元メモ](2026-10-24-dfine-prefix-investigation.md)にある矩形の最終出力不一致を、今回も再現できたとはしない。

モデル・重み・入力 hash・許容差は変更していない。過去の生ログと実行環境が失われているため、この差の理由は特定していない。今回の結果で過去の失敗を消したり、モデルが修正済みだと判断したりしない。

## 実行方法と成否

README の固定依存を専用の detector venv に導入し、既存の `prepare_detector.py` で公開モデルを準備した後に実行する。既存の準備は元の中間特徴量ゲートで非0終了し、調査用資産を保持する。

```bash
artifacts/p1b/.venv-detector/bin/python -B scripts/reproduce_detector_rectangle.py
```

この診断の終了コード0は全測定の完了を意味する。数値結果は `sourceParityPassed` と各比較に保存し、`qualifiesDetector` は常に false。出典・shape・dtype・非有限値・実行例外では非0終了する。元の `prepare_detector.py` の厳格な資格判定は変更していない。

生の画像・tensor・モデルは git に入れず、小さい比較 JSON だけを保存する。Python は開発者の公開生成画像に対する診断に使い、利用者の学習や変換へ要求しない。

## 今回の結果

全出力で元の要素別許容差を使う。追加の normwise 契約や query の並べ替えは使用していない。

| 出力 | 不合格要素数 | 最大絶対差 | 最大許容差比 |
| --- | ---: | ---: | ---: |
| logits | 0 / 24,000 | 2.01702e-4 | 0.2660 |
| boxes | 0 / 1,200 | 2.60472e-5 | 0.3952 |
| bboxInput | 7 / 38,400 | 5.62906e-4 | 1.7031 |
| classInput | 6 / 38,400 | 5.42641e-4 | 1.5960 |
| residualCorners | 245 / 39,600 | 1.09692e-2 | 11.2730 |
| references | 0 / 1,200 | 1.60933e-5 | 0.2925 |

encoder の projection / LayerNorm / box MLP の入力・出力も比較した。今回はすべて元の feature 許容差内だった。ONNX が計算した同じ入力を元の各 module に渡すと、projection と box MLP の出力は一致し、LayerNorm の最大絶対差は9.53674e-7だった。中間出力を追加しても、元の prefix の4出力は bit 単位で不変だった。

この結果は、今回観測した encoder 部品に局所的不一致の証拠がないことを示す範囲に限る。decoder 内で中間特徴の差がどこから許容差外になるか、実用精度への影響は未解明である。

## 範囲

- 既知の合成矩形1画像、固定 COCO-only D-FINE Nano、CPU1スレッド、ORT_ENABLE_ALL。
- 実行済みの6入力を未使用画像として再評価したものではない。
- ブラウザ再現、query 順序、全勾配・更新、実画像品質、WebGPU、端末性能の新しい認定ではない。
- 次は今回保存した入力・モデルを起点に decoder の差を絞る。過去の失われた部品比較を再実行済みと扱わない。

## コードと回帰の確認

固定入力 hash、数値不一致の保持、shape/dtype/非有限値、古い completed の無効化、割込み、出力先保護の6テストが通った。実装前には対象モジュール未存在による失敗を確認した。既存分を含む Python テストは47件成功した。独立レビューで Critical / Important の指摘はなかった。モデル・ブラウザ側の TypeScript は変更していない。

CI はこの診断を既存の検出準備に続く通常ステップとして実行する。診断の数値 FAIL は表に明示し、source/構造/実行エラーはステップ自体を失敗させる。追加の `continue-on-error` は設けない。
