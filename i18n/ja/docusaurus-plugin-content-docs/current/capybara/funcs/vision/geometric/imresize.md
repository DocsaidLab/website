# imresize

> [imresize(img: np.ndarray, size: tuple[int | None, int | None], interpolation: str | int | INTER = INTER.BILINEAR, return_scale: bool = False)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **説明**：入力画像をリサイズします。

- **パラメータ**

  - **img** (`np.ndarray`)：リサイズする入力画像。
  - **size** (`tuple[int | None, int | None]`)：リサイズ後のサイズ（`(height, width)`）。片方が `None` の場合、アスペクト比を維持するようにもう一方が推定されます。
  - **interpolation** (`str | int | INTER`)：補間方法。`INTER.NEAREST` / `INTER.BILINEAR` / `INTER.CUBIC` / `INTER.AREA` / `INTER.LANCZOS4`。デフォルトは `INTER.BILINEAR`。
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
