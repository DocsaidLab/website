---
sidebar_position: 3
---

# imrotate

> [imrotate(img: np.ndarray, angle: float, scale: float = 1, interpolation: Union[str, int, INTER] = INTER.BILINEAR, bordertype: Union[str, int, BORDER] = BORDER.CONSTANT, bordervalue: Union[int, Tuple[int, int, int]] = None, expand: bool = True, center: Tuple[int, int] = None) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L80C1-L153C1)

- **説明**：入力画像に対して回転処理を行います。

- 引数

  - **img** (`np.ndarray`)：回転処理を行う入力画像。
  - **angle** (`float`)：回転角度。度単位で、反時計回りの方向。
  - **scale** (`float`)：スケーリング比率。デフォルトは 1。
  - **interpolation** (`Union[str, int, INTER]`)：補間方法。使用可能なオプションは INTER.NEAREST、INTER.LINEAR、INTER.CUBIC、INTER.LANCZOS4 です。デフォルトは INTER.LINEAR です。
  - **bordertype** (`Union[str, int, BORDER]`)：境界タイプ。使用可能なオプションは BORDER.CONSTANT、BORDER.REPLICATE、BORDER.REFLECT、BORDER.REFLECT_101 です。デフォルトは BORDER.CONSTANT です。
  - **bordervalue** (`Union[int, Tuple[int, int, int]]`)：境界のパディング値。bordertype が BORDER.CONSTANT の場合にのみ有効です。デフォルトは None です。
  - **expand** (`bool`)：回転後の画像全体を収めるために出力画像を拡大するかどうか。True の場合、回転後の画像全体を収めるために出力画像を拡大します。False または省略した場合、出力画像は入力画像と同じサイズになります。expand フラグは、画像が中心を中心に回転し、平行移動しないことを前提としています。デフォルトは False です。
  - **center** (`Tuple[int, int]`)：回転中心。デフォルトは画像の中心です。

- **返り値**

  - **np.ndarray**：回転後の画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  rotate_img = D.imrotate(img, 45, bordertype=D.BORDER.CONSTANT, expand=True)

  # 回転後の画像を元のサイズにリサイズして可視化
  rotate_img = D.imresize(rotate_img, [img.shape[0], img.shape[1]])
  ```

  ![imrotate](./resource/test_imrotate.jpg)
