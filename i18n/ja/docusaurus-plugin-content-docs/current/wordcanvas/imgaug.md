---
sidebar_position: 5
---

# 画像の拡張

私たちは、画像拡張の機能を `WordCanvas` に組み込んでいません。なぜなら、これは非常に「カスタマイズ性の高い」ニーズであり、異なる応用シナリオごとに異なる拡張方法が必要になるからです。しかし、いくつかの簡単なサンプルを提供し、画像拡張のフローをどのように実現するかを説明しています。

画像拡張を実現するために、[**albumentations**](https://github.com/albumentations-team/albumentations) というライブラリをよく使用しますが、お好みのライブラリを使うことも可能です。

:::info
`albumentations` が v2.0.0 にアップデートされた後、多くの操作のパラメータ名が変更されましたのでご注意ください。

関連情報は、[**albumentations v2.0.0**](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0) を参照してください。
:::

## サンプル 1：せん断変換

生成したテキスト画像にカスタム操作を適用します。

まず、せん断変換を適用する例を示します。ここでは `Shear` を例に取ります。

`Shear` クラスは画像にせん断変換を適用します。せん断は画像の幾何形状を変化させ、水平方向の傾斜を作り出します。これにより、モデルが異なる方向や位置で対象を認識する能力を学習するのに役立ちます。

- **パラメータ**

  - max_shear_left: 左方向への最大せん断角度。デフォルトは 20 度。
  - max_shear_right: 右方向への最大せん断角度。同じくデフォルトは 20 度。
  - p: 操作の確率。デフォルトは 0.5 で、任意の画像が 50%の確率でせん断されます。

- **使用方法**

  ```python
  from wordcanvas import Shear, WordCanvas

  gen = WordCanvas()
  shear = Shear(max_shear_left=20, max_shear_right=20, p=0.5)

  img = gen('Hello, World!')
  img = shear(img)
  ```

  ![shear_example](./resources/shear_example.jpg)

## サンプル 2：回転変換

回転変換を実装するには、`albumentations` の `SafeRotate` クラスを使用します。

`Shift`、`Scale`、`Rotate` 関連の操作を使用する場合、背景色の塗りつぶしに関する問題が発生します。

このとき、`infos` 情報を呼び出して背景色を取得する必要があります。

```python
import cv2
from wordcanvas import ExampleAug, WordCanvas
import albumentations as A

gen = WordCanvas(
    background_color=(255, 255, 0),
    text_color=(0, 0, 0),
    return_infos=True
)

img, infos = gen('Hello, World!')

aug = A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    fill=infos['background_color'],
    p=1
)

img = aug(image=img)['image']
```

![rotate_example](./resources/rotate_example.jpg)

## サンプル 3：クラス動作の変更

コードを書き進めるうちに、以下の点に気付くかもしれません：

- 毎回ランダムな背景色で画像を生成する場合、毎回 `albumentations` のクラスを再初期化する必要があるのは効率的ではないのでは？

もしかすると、`albumentations` の動作を変更し、一度初期化するだけで使い続けられるようにできるかもしれません。

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import RandomWordCanvas

gen = RandomWordCanvas(
    random_background_color=True,
    return_infos=True
)

aug = A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    p=1
)

imgs = []
for _ in range(8):
    img, infos = gen('Hello, World!')

    # albumentations クラスの動作を変更
    aug.fill = infos['background_color']

    img = aug(image=img)['image']

    imgs.append(img)

# 結果を表示
img = np.concatenate(imgs, axis=0)
```

![bgcolor_example](./resources/bgcolor_example.jpg)

:::danger
例 2 の方法を使用することを推奨します（たとえ少し冗長に見えても）。`albumentations` クラスの動作を直接変更すると、マルチスレッド環境で問題が発生する可能性がありますので、十分に注意してください。
:::

## サンプル 4：背景の追加

単なるテキスト画像では満足できず、背景を追加してモデルの汎化能力を向上させたい場合があります。

その場合、背景画像を事前に用意し、以下の例を参考にしてください：

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import RandomWordCanvas
from albumentations import RandomCrop

gen = RandomWordCanvas(
    random_text_color=True,
    random_background_color=True,
    return_infos=True
)

# ランダムな色のテキスト画像を生成
img, infos = gen('Hello, World!')
```

![sample25](./resources/sample25.jpg)

次に、背景画像をロードします：

```python
bg = cv2.imread('path/to/your/background.jpg')
```

[![bg_example](./resources/bg_example.jpg)](https://www.lccnet.com.tw/lccnet/article/details/2274)

最後に、背景画像からランダムに領域を切り取り、テキスト画像を重ねます：

```python
bg = RandomCrop(img.shape[0], img.shape[1])(image=bg)['image']

result_img = np.where(img == infos['background_color'], bg, img)
```

![bg_result](./resources/sample26.jpg)

## サンプル 5：透視変換

透視変換は、画像を新しい視平面に投影する変換で、異なる角度や距離での外観をシミュレートできます。

前の例を引き継ぎ、画像に透視変換を適用した後で背景を重ねます：

```python
from albumentations import Perspective

aug = A.Perspective(
    keep_size=True,
    fit_output=True,
    fill=infos['background_color'],
)

img = aug(image=img)['image']
result_img = np.where(img == infos['background_color'], bg, img)
```

![sample27](./resources/sample27.jpg)

:::tip
「空間変化」の画像拡張操作については、まず元の画像に透視変換を適用し、その後で背景画像を重ねることを推奨します。これにより、背景画像に奇妙な黒い縁が生じることを防げます。
:::

## サンプル 6：強光反射

一般的なテキスト画像では、強光反射の問題が発生することがあります。この場合、`RandomSunFlare` を使用してこの状況をシミュレートできます：

```python
from albumentations import RandomSunFlare

aug = A.RandomSunFlare(
    src_radius=128,
    src_color=(255, 255, 255),
)

result_img = aug(image=result_img)['image']
```

![sample28](./resources/sample28.jpg)

:::tip
「ピクセル変化」の画像拡張操作については、背景画像を重ねた後で画像拡張変換を行うことを推奨します。これにより、背景情報が失われて雑多な斑点が生じるのを防げます。
:::

## 結論

本プロジェクトの紹介は以上で終了です。ご質問やご提案がありましたら、下にコメントを残してください。できるだけ早く返信いたします。

また、特定の操作をどのように実現するか分からない場合も、コメントを残していただければ、できる限りサポートします。

楽しくご利用ください！
