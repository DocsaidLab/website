# DocAligner デモ

ファイルシステムからいくつかの文書画像を選んで、この機能をテストしてみてください。

もし画像が見つからない場合は、MIDV-2020 からいくつか借りて試すこともできます：

:::info
**下の画像をクリックすると、Demo 画像に直接代入できます。**

これらの画像はトレーニングデータに含まれているため、効果が良好です。モデルはすでにこれらを見たことがあります！

実際のアプリケーションでは、さまざまな状況に直面する可能性があるため、異なる画像を使ってテストすることをお勧めします。そうすることで、モデルの効果をより理解できるでしょう。

このウェブページの機能を使用する際、いくつかの注意点があります：

1. **画像外に文書のコーナーがある場合、モデルは 4 つのコーナーを認識できず、エラーメッセージが表示されます。**
   - 未知の領域に対してモデルが外挿できるように努力していますが、それでも失敗することがあります。
2. **画像内に複数の文書が同時に存在する場合、モデルはランダムに 4 つのコーナーを選ぶことがあります。**
3. **ウェブページの負荷を考慮し、画像を圧縮しています。そのため、画像の品質が若干低下します。**
   - これを行わないと、ブラウザがクラッシュする可能性があります。
4. **OpenCV モジュールを非同期的にダウンロードしています。最終的にトリミング画面が空白の場合、ダウンロードが完了していないためです。**
   - OpenCV は大きい（約 8MB）ので、少し待つ必要があります。
   - この機能は必須ではなく、無視しても問題ありません。

以上の注意点をお守りいただき、楽しんでください！
:::

もし自分のプログラムで使用したい場合、推論プログラムのサンプルコードを参考にできます：

```python title='python demo code'
from docaligner import DocAligner
import docsaidkit as D

model = DocAligner(model_cfg='fastvit_sa24')

# 画像内で未知のコーナーを見つけるためのパディング
input_img = D.pad(input_img, 100)

polygon = model(
    img=input_img,
    do_center_crop=False,
    return_document_obj=False
)

# パディングを削除
polygon -= 100

return polygon
```

:::tip
MIDV-2020 はオープンソースのデータセットで、たくさんの文書画像が含まれています。これを使って文書分析モデルをテストできます。

必要に応じて、こちらからダウンロードできます：[**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import DocAlignerDemoWrapper from "@site/src/components/DocAlignerDemoWrapper";

<DocAlignerDemoWrapper
titleStage1="テスト画像"
titleStage2="モデル展示"
chooseFileLabel="ファイルを選択"
uploadButtonLabel="アップロードして予測"
downloadButtonLabel="予測結果をダウンロード"
clearButtonLabel="結果をクリア"
processingMessage="処理中です。しばらくお待ちください..."
errorMessage={{
    chooseFile: "ファイルを選択してください",
    invalidFileType: "JPG、PNG、Webp形式の画像のみ対応しています",
    networkError: "ネットワークエラーです。後でもう一度お試しください。",
    uploadError: "エラーが発生しました。後でもう一度お試しください。",
  }}
warningMessage={{
    noPolygon:
      "4つのコーナーが検出されませんでした。モデルはこの文書タイプを認識していない可能性があります。",
    imageTooLarge:
      "画像が大きすぎます。ブラウザがクラッシュする可能性があります。",
  }}
imageInfoTitle="画像情報"
inferenceInfoTitle="モデル推論情報"
polygonInfoTitle="検出結果"
inferenceTimeLabel="推論時間"
timestampLabel="タイムスタンプ"
fileNameLabel="ファイル名"
fileSizeLabel="ファイルサイズ"
fileTypeLabel="ファイルタイプ"
imageSizeLabel="画像サイズ"
TransformedTitle="平坦化画像"
TransformedWidthLabel="出力幅"
TransformedHeightLabel="出力高さ"
TransformedButtonLabel="平坦化画像をダウンロード"
defaultImages={[
{ src: "/ja/img/docalign-demo/000025.jpg", description: "文字干渉" },
{ src: "/ja/img/docalign-demo/000121.jpg", description: "部分的な隠れ" },
{ src: "/ja/img/docalign-demo/000139.jpg", description: "強い反射" },
{ src: "/ja/img/docalign-demo/000169.jpg", description: "暗いシーン" },
{ src: "/ja/img/docalign-demo/000175.jpg", description: "強い歪み" },
]}
/>
