---
sidebar_position: 5
---

# 圖像增強

我們沒有把圖像增強的功能做在 `WordCanvas` 內，因為我們認為這是一個非常個性化的需求，不同的應用場景可能需要不同的增強方式。但我們提供了一些簡單的範例，說明該如何實現圖像增強的流程。

我們習慣使用 [**albumentations**](https://github.com/albumentations-team/albumentations) 這個套件來實現圖像增強，但是你可以使用任何你喜歡的庫。

## 範例一：`Shear`

生成文字圖像後，再套用自定義的操作，以下以 `Shear` 為例。

`Shear` 類負責對圖像進行剪切變換。剪切會改變圖像的幾何形狀，創造出水平的傾斜，這可以幫助模型學習在不同的方向和位置識別對象。

- **參數**
    - max_shear_left：向左的最大剪切角度。默認值為 20 度。
    - max_shear_right：向右的最大剪切角度。默認值同樣為 20 度。
    - p：操作概率。默認為 0.5，意味著任何給定的圖像有 50% 的概率會被剪切。

- **使用方式**

    ```python
    from wordcanvas import Shear, WordCanvas

    gen = WordCanvas()
    shear = Shear(max_shear_left=20, max_shear_right=20, p=0.5)

    img, _ = gen('Hello, World!')
    img = shear(img)
    ```

    ![shear_example](./resources/shear_example.jpg)

##  範例二：`SafeRotate`

使用 `Shift`、`Scale`或 `Rotate` 相關的操作時，會遇到背景色填充的問題。

這時應該調用 `infos` 資訊來獲取背景色。

```python
from wordcanvas import ExampleAug, WordCanvas
import albumentations as A

gen = WordCanvas(
    background_color=(255, 255, 0),
    text_color=(0, 0, 0)
)

aug =  A.SafeRotate(
    limit=30,
    border_mode=cv2.BORDER_CONSTANT,
    value=infos['background_color'],
    p=1
)

img, infos = gen('Hello, World!')
img = aug(image=img)['image']
```

![rotate_example](./resources/rotate_example.jpg)

## 範例三：修改 albu 類別行為

程式寫到這裡，你可能會注意到：

- 如果每次 `WordCanvas` 生成圖像都帶有隨機背景色，那麼每次都需要重新初始化 `albumentations` 的類別，是不是不科學？

答對了！所以我們要修改 `albumentations` 的行為，讓它只需要一次初始化就可以一直使用。

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

    # 修改 albu 類別行為
    aug.value = infos['background_color']

    img = aug(image=img)['image']

    imgs.append(img)

# 顯示結果
img = np.concatenate(imgs, axis=0)
```

![bgcolor_example](./resources/bgcolor_example.jpg)

## 結語

在本專案中，我們只專注於對影像進行增強，而無需調整包含邊界框（bounding box）和遮罩（mask）的複雜元素。

如果你有任何問題或建議，歡迎在底下留言，我們會盡快回覆。

本專案的介紹到此結束，祝你使用愉快！
