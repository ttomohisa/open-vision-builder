# D-FINE own-prefix 影響測定: 固定する実画像データ

- 日付: 2026-10-25 UTC。
- 用途: [ADR 0005](../decisions/0005-bound-numerical-investigation.md) の限定比較で使う COCO val2017 の10画像。
- 正本: `tests/fixtures/detector-impact-data.json`。全58 instance annotation、元の画像メタデータ、80クラス対応、出典、著作者、ライセンス、画像・入力・分割の hash を保持する。
- このデータ準備ではモデルの推論・学習・品質測定を実行していない。元の G1 FAIL や検出器の資格判定は変更しない。

## 選定をモデル測定前に固定する

COCO image ID の昇順に、次の条件を満たす最初の10画像を選んだ。最初の4画像を train、次の2画像を tuning、最後の4画像を evaluation とする。候補の採否にモデル出力を使っていない。

1. COCO の画像ライセンスが id=4、`Attribution License`、CC BY 2.0 である。
2. annotation が1〜12個あり、画像内に `iscrowd=1` の annotation が一つもない。
3. 全 bbox が有限・正の幅と高さを持ち、元画像の範囲内である。浮動小数点表記の境界には `1e-5` pixel の確認余裕だけを使う。
4. 原画像の Flickr ページから著作者・作品名を取得でき、現在の photo license も CC BY 2.0 と確認できる。

採用前の119候補の除外理由は manifest の `selection.excludedCandidates` に保存した。例えば image 285 は COCO 上では CC BY 2.0 だが、2026-10-25 の Flickr ページは All rights reserved だったため除外した。元の CC ライセンスの法的な存続を判断するための実験には広げず、現在の表示も一致する候補に限定した。削除済み等で photo license を取得できない候補も、その観測理由を記録して除外した。

採用画像のすべての COCO annotation を保持し、主対象以外のクラスも落としていない。独自に bbox を追加・修正したり、予測と合わせたりしていない。COCO 自体の注釈漏れがないことを保証するものではない。

## 出典・利用条件・著作者

[COCO 公式 Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) は annotation を COCO Consortium の **CC BY 4.0** とし、画像の著作権を COCO Consortium が所有しないことを明記する。画像は一括して CC BY 4.0 と扱わず、下表の各 Flickr 写真の **CC BY 2.0** を記録した。[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) と [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) は商用の共有・改変を認め、著作者表示・ライセンスへのリンク等を要求する。

すべての原 JPEG はそのまま保持する。実験入力は RGB 化、640×640 への bilinear stretch、RGB/255、NCHW FP32 化を行う。この変更内容も各画像の `attribution.changes` に記録した。原画像やその派生画像を再配布する場合は、各著作者・作品名・元ページ・CC BY 2.0 の表示と、annotation の COCO Consortium / CC BY 4.0 の表示を同梱する。

| COCO ID | 分割 | 写真 / 著作者 | 全 bbox 数 |
| --- | --- | --- | ---: |
| 000000000785 | train | [Han — Nick Webb](https://www.flickr.com/photos/nickwebb/6795644157/) | 2 |
| 000000000885 | train | [US Open 2013 — Edwin Martinez](https://www.flickr.com/photos/rhythmstrip/9639200419/) | 9 |
| 000000001268 | train | [IMG_4362 — Marko Kudjerski](https://www.flickr.com/photos/marko8904/9413846304/) | 11 |
| 000000002473 | train | [Mt. Washington — Moosealope](https://www.flickr.com/photos/8510057@N02/4377710907/) | 5 |
| 000000006012 | tuning | [Fruit Logo — Erich Ferdinand](https://www.flickr.com/photos/erix/4281955635/) | 2 |
| 000000006818 | tuning | [squat toilet/2nd bathroom — momovieman](https://www.flickr.com/photos/11388848@N07/2068039201/) | 1 |
| 000000007281 | evaluation | [P1020024 — Phil_Heck](https://www.flickr.com/photos/kleche/5022207571/) | 12 |
| 000000008762 | evaluation | [Stoplight on AZ-90, Sierra Vista on the horizon — Patrick Finnegan](https://www.flickr.com/photos/vax-o-matic/4466652187/) | 7 |
| 000000012667 | evaluation | [Monkeyphone At The Office — Erich Ferdinand](https://www.flickr.com/photos/erix/499050773/) | 1 |
| 000000013291 | evaluation | [1_DSC0041 — Johnny Ashburn](https://www.flickr.com/photos/reaperstyle/3491087208/) | 8 |

Flickr ページの取得日・取得本文の SHA-256 と現在の著作者表記を manifest に保持する。動的なページ本文を再取得したときの一致は要求しない。固定された画像 bytes、出典・権利情報、annotation、入力を再現するための記録である。

## 取得元と固定 hash

[COCO 公式 Download のソース](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm) に掲載された archive は `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`。本環境では `https://images.cocodataset.org/` は503を返したため、同じ公開 bucket の TLS endpoint `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip` を使った。両 endpoint の HEAD は同じ252,907,541 bytes と ETag `f4bbac642086de4f52a3fdda2de5fa2c` を返すことを確認した。archive は一度だけ取得し、`annotations/instances_val2017.json` を取り出した。

| 固定対象 | SHA-256 |
| --- | --- |
| `annotations_trainval2017.zip` (252,907,541 bytes) | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `instances_val2017.json` (19,987,840 bytes) | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |
| 固定 manifest | `464570f5138ddfdd159892a9e3a9a4723d45ee335de1bc93965bc93fc6b289a6` |
| ID と分割の canonical JSON | `59efc54b19cc824da2f640ac2a4553c532cfc82c1546da3eaacf0bfd382807c0` |
| ID と元の全 annotation の canonical JSON | `dc6229794d77f91d5fc359a93cf1da244752f2f75e785a1251e0eb8ffb9be461` |

canonical JSON の定義は UTF-8、`sort_keys=True`、`separators=(',', ':')`、`ensure_ascii=False`。画像と入力ごとの SHA-256、byte数、元画像の幅・高さは manifest にある。10原画像は計1,158,262 bytes。入力は1画像4,915,200 bytes、計49,152,000 bytes。

画像の取得 URL は `https://s3.amazonaws.com/images.cocodataset.org/val2017/{12桁ID}.jpg` に固定する。全画像 zip は取得しない。再現時は committed manifest の annotation を使うので、250 MB の annotation archive を再度取得する必要もない。最初の archive、抽出 JSON、Flickr 確認資料は ignored の `artifacts/p1d/data/source/` にある。

## 前処理とラベル対応

- Pillow **11.1.0**、NumPy **2.2.3** を使用する。
- 原画像の decode サイズが COCO width/height と一致することを確認する。EXIF 回転は追加しない。
- `image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR)`。全画面の stretch で、crop・padding はない。
- uint8 の RGB を float32 に変換して NCHW にし、`np.float32(255)` で割る。little-endian FP32、shape `[1,3,640,640]` として保存する。
- 同じバイナリを PyTorch と実ブラウザの両方で読む。ブラウザ側で JPEG を再 decode / resize して入力差を追加しない。
- COCO bbox `[x,y,w,h]` を元画像の幅 `W` / 高さ `H` で `[(x+w/2)/W,(y+h/2)/H,w/W,h/H]` にする。箱の順番は元 annotation ID の昇順。
- 80カテゴリの疎な COCO category ID を昇順に並べ、その位置を0〜79の学習ラベルにする。既存 `artifacts/p1b/detector-assets/unqualified/model.json` の全80 label と照合した。

表記だけが異なる6クラスを明示して対応付け、それ以外は名前の完全一致を要求する。

| COCO | モデルの label |
| --- | --- |
| motorcycle | motorbike |
| airplane | aeroplane |
| couch | sofa |
| potted plant | pottedplant |
| dining table | diningtable |
| tv | tvmonitor |

## 実行と検証

```bash
artifacts/p1b/.venv-detector/bin/python scripts/prepare_detector_impact_data.py
artifacts/p1b/.venv-detector/bin/python -m unittest discover -s tests -p 'detector_impact_data_test.py'
```

準備先は `artifacts/p1d/data/`。`images/` に元 JPEG、`inputs/` に canonical tensor、`prepared.json` に manifest hash を保存する。コードは変更可能な選定器を含まず、committed manifest の10画像だけを再現する。未知の URL / path、symlink、無関係な既存ファイル、重複画像、誤った分割、改変された箱・ラベル・hash を拒否する。全画像と tensor の hash 確認が終わるまで prepared データを公開しない。再実行は既存ファイルの hash を再検証し、上書きや再ダウンロードを行わない。

初回準備と、ネットワーク権限を使わない再実行の両方で10画像・4/2/4分割を確認した。最初に未実装の helper に対する8テストの失敗を確認して実装した。その後、split / annotation hash の改変を拒否できない失敗を確認して追加検証を実装した。最終的にこのデータ専用の **11テストが成功**した。テストは正規化・不正 bbox・不明カテゴリ・分割重複・未知 URL / path・hash 改変・hash 不一致時の非公開・symlink / 既存ファイル保護・RGB/NCHW変換を対象とする。

## この分割の意味と限界

evaluation は**今回の追加学習・調整に未使用の4画像**を意味する。元の COCO 学習済み checkpoint のモデル選定や上流開発で未使用だったとは保証しない。既知の6つの合成・回帰入力は含めていない。この10画像を一般的な検出品質・独自クラス・全層 fine-tuning の認定には使わない。

原画像と canonical 入力の hash は全10画像で異なる。測定前の contact sheet 目視では同じ写真の重複は見当たらなかった。一方、tuning 6012 と evaluation 12667 は同じ著作者による別のバナナの構図であり、この split は**画像単位の分離**で、著作者単位の分離ではない。クラスごとのバランスや代表性を最適化した subset でもない。4枚の結果は少数例での runtime 比較として扱い、一般化性能の推定へ広げない。評価結果を見て画像・分割・annotation・前処理を交換しない。
