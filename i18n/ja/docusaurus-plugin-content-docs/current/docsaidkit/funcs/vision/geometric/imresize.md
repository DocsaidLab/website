---
sidebar_position: 1
---

# imresize

> [imresize(img: np.ndarray, size: Tuple[int, int], interpolation: Union[str, int, INTER] = INTER.BILINEAR, return_scale: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L15)

- **説明**：入力画像をリサイズします。

- 引数

  - **img** (`np.ndarray`)：リサイズする入力画像。
  - **size** (`Tuple[int, int]`)：リサイズ後の画像のサイズ。1 つの次元のみが指定された場合、元の画像のアスペクト比を保ちながら他の次元を計算します。
  - **interpolation** (`Union[str, int, INTER]`)：補間方法。使用可能なオプションは INTER.NEAREST、INTER.LINEAR、INTER.CUBIC、INTER.LANCZOS4 です。デフォルトは INTER.LINEAR です。
  - **return_scale** (`bool`)：スケール比を返すかどうか。デフォルトは False。

- **返り値**

  - **np.ndarray**：リサイズ後の画像。
  - **Tuple[np.ndarray, float, float]**：リサイズ後の画像と、幅と高さのスケール比。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')

  # 画像を H=256, W=256 にリサイズ
  resized_img = D.imresize(img, [256, 256])

  # 画像を H=256 にリサイズし、アスペクト比を維持
  resized_img = D.imresize(img, [256, None])

  # 画像を W=256 にリサイズし、アスペクト比を維持
  resized_img = D.imresize(img, [None, 256])
  ```
