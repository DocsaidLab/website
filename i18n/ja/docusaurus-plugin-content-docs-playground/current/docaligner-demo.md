# DocAligner Demo

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
from capybara import pad

model = DocAligner(model_cfg='fastvit_sa24')

# 画像内で未知のコーナーを見つけるためのパディング
input_img = pad(input_img, 100)

polygon = model(
  img=input_img,
  do_center_crop=False
)

# パディングを削除
polygon -= 100

return polygon
```

:::tip
MIDV-2020 はオープンソースのデータセットで、たくさんの文書画像が含まれています。これを使って文書分析モデルをテストできます。

必要に応じて、こちらからダウンロードできます：[**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo';
import demoContent from '@site/src/data/demoContent';

export function Demo() {
const currentLocale = 'ja';
const localeContent = demoContent[currentLocale];
return <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />;
}

<Demo />
