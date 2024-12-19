---
sidebar_position: 6
---

> [imbinarize(img: np.ndarray, threth: int = cv2.THRESH_BINARY, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L336)

- **説明**：入力画像に対して二値化処理を行います。

- 引数

  - **img** (`np.ndarray`)：二値化処理を行う入力画像。入力画像が 3 チャンネルの場合、関数は自動的に `bgr2gray` 関数を適用します。
  - **threth** (`int`)：閾値タイプ。2 種類の閾値タイプがあります：
    1. `cv2.THRESH_BINARY`：`cv2.THRESH_OTSU + cv2.THRESH_BINARY`
    2. `cv2.THRESH_BINARY_INV`：`cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV`
  - **color_base** (`str`)：入力画像の色空間。デフォルトは `'BGR'`。

- **返り値**

  - **np.ndarray**：二値化後の画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  bin_img = D.imbinarize(img)
  ```

  ![imbinarize](./resource/test_imbinarize.jpg)
