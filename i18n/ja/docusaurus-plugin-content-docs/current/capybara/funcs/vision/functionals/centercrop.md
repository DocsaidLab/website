---
sidebar_position: 8
---

# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/functionals.py#L374)

- **説明**：入力画像に対してセンタークロップ処理を行います。

- **引数**

  - **img** (`np.ndarray`)：センタークロップ処理を行う入力画像。

- **返り値**

  - **np.ndarray**：クロップ後の画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  img = D.imresize(img, [128, 256])
  crop_img = D.centercrop(img)
  ```

  緑色の枠がセンタークロップされた領域を示しています。

  ![centercrop](./resource/test_centercrop.jpg)
