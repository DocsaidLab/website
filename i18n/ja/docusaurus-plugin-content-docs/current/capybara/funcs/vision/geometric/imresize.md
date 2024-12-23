# imresize

> [imresize(img: np.ndarray, size: Tuple[int, int], interpolation: Union[str, int, INTER] = INTER.BILINEAR, return_scale: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L15)

- **説明**：入力画像をリサイズします。

- **パラメータ**

  - **img** (`np.ndarray`)：リサイズする入力画像。
  - **size** (`Tuple[int, int]`)：リサイズ後の画像のサイズ。1 つの次元のみ指定された場合、元の画像のアスペクト比を保持してもう一方の次元が計算されます。
  - **interpolation** (`Union[str, int, INTER]`)：補間方法。選べるオプションは、INTER.NEAREST、INTER.LINEAR、INTER.CUBIC、INTER.LANCZOS4 です。デフォルトは INTER.LINEAR です。
  - **return_scale** (`bool`)：スケールを返すかどうか。デフォルトは False です。

- **戻り値**

  - **np.ndarray**：リサイズ後の画像。
  - **Tuple[np.ndarray, float, float]**：リサイズ後の画像と幅・高さのスケール比率。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')

  # 画像を高さ256、幅256にリサイズ
  resized_img = cb.imresize(img, [256, 256])

  # 画像を高さ256にリサイズし、アスペクト比を保持
  resized_img = cb.imresize(img, [256, None])

  # 画像を幅256にリサイズし、アスペクト比を保持
  resized_img = cb.imresize(img, [None, 256])
  ```
