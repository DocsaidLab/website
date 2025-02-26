# MRZScanner Demo

MRZ が含まれた画像をファイルシステムから選んで、この機能をテストすることができます。

ただし、一般的にはパスポートを持っていない限り、MRZ が含まれる画像は見つけにくいでしょう。（😀 😀 😀）

心配無用、今回はいつものように MIDV-2020 からいくつか借りてきます！

:::info
**下の画像をクリックすると、Demo に直接読み込んでテストできます。**

MIDV-2020 には MRZ の領域アノテーションがないため、これらのデータはモデルが未学習です。

実際のアプリケーションでは、スマホで撮影した写真にはさまざまなケースが考えられます。異なる画像を使ってテストを行い、モデルの効果を比較することをお勧めします。

このウェブページの機能を使用する際、以下の注意点があります：

1. **MRZ 領域が不完全または存在しない場合、モデルは適当な場所を囲みます。**
2. **画像に複数の MRZ が含まれている場合、モデルはランダムに 4 つの点を選択します。**
3. **ウェブブラウザの負荷制限のため、画像を圧縮していますので、画像の品質は低下します。**
   - 圧縮しないと、ブラウザがクラッシュする恐れがあります。

最後に、`DocAligner Demo`の機能をバックエンドで統合しました。`do_doc_align`を有効にするだけで、スムーズに接続できます。

以上の点をご了承ください、楽しんでください！
:::

プログラムから呼び出したい場合、以下の簡単な Python デモコードを参考にしてください：

```python title='python demo code'
from mrzscanner import MRZScanner, ModelType

model = MRZScanner(
   model_type=ModelType.two_stage,
   detection_cfg='20250222',
   recognition_cfg='20250221'
)

result = model(
    img=input_img,
    do_center_crop=False,   # 中心裁切を行うかどうか
    do_postprocess=True     # 後処理（MRZ文字の修正）を行うかどうか
)

return result
```

:::tip
MIDV-2020 はオープンソースのデータセットで、さまざまな文書画像が含まれており、ドキュメント分析モデルのテストに使用できます。

必要な場合、こちらからダウンロードできます：[**MIDV-2020 ダウンロード**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import MRZScannerDemoWrapper from '@site/src/components/MRZScannerDemo';
import mrzdemoContent from '@site/src/data/mrzdemoContent';

export function Demo() {
const currentLocale = 'ja';
const localeContent = mrzdemoContent[currentLocale];
return <MRZScannerDemoWrapper {...localeContent.mrzScannerProps} />;
}

<Demo />
