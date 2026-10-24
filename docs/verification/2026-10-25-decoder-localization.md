# 矩形入力での decoder 数値差の切り分け

実行日: 2026-10-25 JST。起点: PR #5 の `f1aa65498d6e381ef31b972cc0275a40acec618d`。

**観測した境界では、最初の corner MLP が許容差内の入力差を増幅し、出力130要素で元の数値条件を超えた。同じ ONNX 入力を元の module に渡すと、この MLP の出力は bit 単位で一致した。** 3つの decoder layer と3つの MLP の同一入力比較はすべて許容差内だった。元の G1 と矩形の6出力全体は FAIL のままである。

[コード](../../scripts/diagnose_detector_decoder.py)・[入口](../../scripts/reproduce_detector_rectangle.py)・[測定 JSON](evidence/2026-10-25-decoder-cpu.json)を保存する。過去の失われた調査の復元ではなく、[前回保存した矩形再現](2026-10-25-rectangle-reproduction.md)から新しく実行した結果である。

## 条件と再実行

前回と同じ固定 COCO-only D-FINE Nano、公開重み、640×640 の合成矩形1画像、FP32、CPU1スレッド、ORT_ENABLE_ALL を使用した。元の6出力の許容差は変更しない。追加した特徴境界も atol=2e-4 / rtol=1e-4、実際に消費する reference 座標は atol=2e-5 / rtol=1e-4 と実行前に記録した。

README の detector 専用 venv と準備済みの調査用資産を使う。

```bash
artifacts/p1b/.venv-detector/bin/python -B scripts/reproduce_detector_rectangle.py \
  --output artifacts/p1c/rectangle-decoder.json
artifacts/p1b/.venv-detector/bin/python -B -m unittest discover -s tests -p '*_test.py'
```

診断は終了0で完了し、`sourceParityPassed=false` / `qualifiesDetector=false` を保存した。Python テストは53件成功。CI は既存の矩形診断コマンドで同じ追加観測を行い、数値判定と処理完了を分けて表示する。

## 比較しているもの

- **それぞれの実行での出力比較:** 元モデルと ONNX がそれぞれ計算した入力を使うため、上流からの差も含む。
- **同一 ONNX 入力での再計算:** ONNX が実際に渡した全動的入力を、変更していない元の PyTorch module に渡す。その部品内で生じる差を調べる。

decoder layer の再計算には hidden、clamp 後の position embedding、その段が消費する reference、raw encoder memory を使う。memory は query 選択用の正規化済み特徴とは別である。feature の高さ・幅も ONNX 内の4つの整数 scalar から取得し、native 側の `[[40,40],[20,20]]` と照合した。mask なし・attention 出力なしの同じ eval 経路を使う。

hook では配列をコピーし、再計算前に追加 hook をすべて外す。ONNX に中間出力を追加しても、元の prefix の4出力はすべて bit 単位で不変だった。追加した graph の結果で元の計算を置き換えていない。

### 固定 anchor と更新後の reference

既存の `references` 出力は、layer 0 の後に pre-bbox MLP から作った**固定 anchor**である。各段の corner 分布から更新され、次の layer が実際に消費する reference とは異なる。今回は3段の消費 reference を直接比較し、すべて元の座標許容差内だった。

## 実測結果

layer の番号はコードに合わせて0始まり。corner は位置補正の分布 logits を出す MLP であり、その絶対差は最終 box 座標の差とは異なる。

| 観測する出力 | 各 runtime 自身の入力での不合格数 | 最大絶対差 | 同一 ONNX 入力での不合格数 | 同一入力での最大絶対差 |
| --- | ---: | ---: | ---: | ---: |
| layer 0 | 0 / 38,400 | 8.64267e-5 | 0 / 38,400 | 1.43051e-6 |
| pre-bbox MLP | 0 / 1,200 | 8.56817e-5 | 0 / 1,200 | 0 |
| corner MLP 0 | 130 / 39,600 | 1.02644e-2 | 0 / 39,600 | 0 |
| layer 1 | 1 / 38,400 | 3.31193e-4 | 0 / 38,400 | 5.84126e-6 |
| corner MLP 1 | 196 / 39,600 | 7.56621e-4 | 0 / 39,600 | 0 |
| layer 2 | 6 / 38,400 | 5.42641e-4 | 0 / 38,400 | 4.17233e-6 |

corner MLP 0 に入る特徴の最大絶対差は8.64267e-5、全38,400要素が許容差内だった。その出力では130要素が許容差を超え、最大許容差比は8.47154だった。同一 ONNX 入力に対しては3つの MLP の全要素が bit 単位で一致した。

query の行は並べ替えていない。ONNX の300個の TopK index で native 側の encoder 特徴を gather すると、native decoder が実際に消費した特徴の並びと bit 単位で一致した。これは**消費された特徴の順序の一致**であり、同じ値の encoder 行が重複する場合まで、観測していない native index の一致を保証するものではない。

元の6出力の不合格数は前回と同じだった: logits 0、boxes 0、bboxInput 7、classInput 6、residualCorners 245、references 0。新しい hook や出力追加で元の不一致を消していない。

## 分かったことと未解決事項

今回の入力では、同一入力での局所比較がすべて通る一方、各 runtime 自身の入力からの比較は一部で失敗した。特に最初の corner MLP の大きな差は、上流から来た異なる入力を同じ MLP が増幅したことで再現できる。MLP の変換ミスを示す観測ではない。

これはモデル全体の数値同等性を証明しない。各 layer 内の小さな差が反復計算で増幅する可能性や、encoder 以前の差の寄与は残る。次は decoder 全体に同じ入口 tensor を与えた比較で、上流の差と decoder 内で蓄積する差を分ける。未測定の結果はここには記載しない。

合成1画像の既知入力だけで、ブラウザ、独自クラス、全学習、未使用実画像品質、WebGPU、メモリを新たに認定したものではない。閾値変更・重み変更・query 再整列・検出 UI への接続は行っていない。
