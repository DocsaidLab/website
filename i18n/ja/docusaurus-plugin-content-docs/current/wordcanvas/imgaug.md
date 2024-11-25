---
sidebar_position: 5
---

# 画像強化

`WordCanvas` 内では画像強化の機能は実装していません。これは、画像強化が非常に「カスタマイズ可能な」ニーズであり、異なるアプリケーションシーンでは異なる強化方法が必要になる可能性があるためです。しかし、画像強化のプロセスを実装する方法について簡単な例をいくつか提供しています。

私たちは [**albumentations**](https://github.com/albumentations-team/albumentations) ライブラリを使用して画像強化を実現するのが一般的ですが、好きなライブラリを使用してもかまいません。

## 例 1: シアー変換

文字画像を生成した後、カスタム操作を適用する方法を示します。

まず、シアー変換を適用する例を示します。`Shear` クラスを使用します。

`Shear` クラスは画像にシアー変換を行います。シアーは画像の幾何学的形状を変え、水平な傾きを作り出します。これにより、モデルが異なる方向と位置でオブジェクトを認識するのを学習するのを助けます。

- **パラメータ**

  - `max_shear_left`: 左への最大シアー角度。デフォルト値は 20 度です。
  - `max_shear_right`: 右への最大シアー角度。デフォルト値は同じく 20 度です。
  - `p`: 操作の確率。デフォルトは 0.5 で、与えられた画像に 50%の確率でシアー変換が適用されます。

- **使用方法**

  ```python
  from wordcanvas import Shear, WordCanvas

  gen = WordCanvas()
  shear = Shear(max_shear_left=20, max_shear_right=20, p=0.5)

  img, _ = gen('Hello, World!')
  img = shear(img)
  ```

  ![shear_example](./resources/shear_example.jpg)

## 例 2: 回転変換

回転変換を実装するために、`albumentations` から `SafeRotate` クラスをインポートします。

`Shift`、`Scale`、`Rotate` に関連する操作を使用する際、背景色の埋め込み問題が発生します。

この場合、`infos` から背景色を取得する必要があります。

```python
from wordcanvas import ExampleAug, WordCanvas
import albumentations as A

gen = WordCanvas(
    background_color=(255, 255, 0),
    text_color=(0, 0, 0)
)

aug = A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    value=infos['background_color'],
    p=1
)

img, infos = gen('Hello, World!')
img = aug(image=img)['image']
```

![rotate_example](./resources/rotate_example.jpg)

## 例 3: クラスの動作を変更

ここまでのコードで気づいたかもしれませんが：

- `WordCanvas` で生成した画像が毎回ランダムな背景色を持っていると、`albumentations` クラスを毎回再初期化する必要があるのは非効率です。

代わりに、`albumentations` の動作を変更して、一度の初期化で繰り返し使用できるようにする方法を示します。

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import WordCanvas

gen = WordCanvas(
    random_background_color=True
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
    aug.value = infos['background_color']

    img = aug(image=img)['image']

    imgs.append(img)

# 結果を表示
img = np.concatenate(imgs, axis=0)
```

![bgcolor_example](./resources/bgcolor_example.jpg)

:::danger
直接 `albumentations` のクラスの動作を変更する方法は、複数スレッドでのトレーニング環境では問題を引き起こす可能性があるため、推奨しません。なるべく `例2` の方法を使用してください。
:::

## 例 4: 背景の追加

単純な文字画像に加えて背景を追加したい場合、モデルの汎化能力を向上させるために背景画像を準備し、以下の例に従ってください。

```python
import albumentations as A
import cv2
import numpy as np
from wordcanvas import WordCanvas
from albumentations import RandomCrop

gen = WordCanvas(
    random_text_color=True,
    random_background_color=True
)

# ランダムな背景色の文字画像を生成
img, infos = gen('Hello, World!')
```

![sample25](./resources/sample25.jpg)

次に、背景画像を読み込みます：

```python
bg = cv2.imread('path/to/your/background.jpg')
```

[![bg_example](./resources/bg_example.jpg)](https://www.lccnet.com.tw/lccnet/article/details/2274)

その後、背景からランダムに切り取った領域に文字画像を重ねます：

```python
bg = RandomCrop(img.shape[0], img.shape[1])(image=bg)['image']

result_img = np.where(img == infos['background_color'], bg, img)
```

![bg_result](./resources/sample26.jpg)

## 例 5: 透視変換

透視変換は、画像を新しい視点に投影する変換で、物体が異なる角度や距離でどう見えるかをシミュレートすることができます。

前の例に続き、透視変換を適用してから背景を合成する方法を示します：

```python
from albumentations import Perspective

aug = A.Perspective(
    keep_size=True,
    fit_output=True,
    pad_val=infos['background_color'],
)

img = aug(image=img)['image']
result_img = np.where(img == infos['background_color'], bg, img)
```

![sample27](./resources/sample27.jpg)

:::tip
「空間変更」の画像強化操作では、透視変換を最初に行ってから背景を追加するのがベストです。こうすることで背景が変な黒い境界を持たないようにできます。
:::

## 例 6: 強い光の反射

通常の文字にも強い光の反射が問題になることがあります。これをシミュレートするために、`RandomSunFlare` を使用します：

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
「ピクセル変更」の画像強化操作では、背景を先に追加し、次に画像強化を行うのが推奨されます。これにより背景情報が失われず、雑な斑点が現れることを避けることができます。
:::

## 結論

このプロジェクトの紹介は以上です。ご質問や提案がある場合は、下記にコメントしてください。できるだけ早くお答えします。

また、特定の操作方法が分からない場合も、コメントをいただければお手伝いできる限りサポートいたします。

ご利用をお楽しみください！
