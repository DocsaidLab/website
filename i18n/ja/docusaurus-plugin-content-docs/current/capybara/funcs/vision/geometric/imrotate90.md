---
sidebar_position: 2
---

# imrotate90

> [imrotate90(img: np.ndarray, rotate_code: ROTATE) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L66C1-L77C47)

- **説明**：入力画像を 90 度回転させます。

- 引数

  - **img** (`np.ndarray`)：回転処理を行う入力画像。
  - **rotate_code** (`RotateCode`)：回転コード。使用可能なオプションは以下の通りです：
    - ROTATE.ROTATE_90：90 度回転。
    - ROTATE.ROTATE_180：180 度回転。
    - ROTATE.ROTATE_270：反時計回りに 90 度回転。

- **返り値**

  - **np.ndarray**：回転後の画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  rotate_img = D.imrotate90(img, D.ROTATE.ROTATE_270)
  ```

  ![imrotate90](./resource/test_imrotate90.jpg)
