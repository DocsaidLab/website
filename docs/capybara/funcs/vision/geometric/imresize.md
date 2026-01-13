# imresize

> [imresize(img: np.ndarray, size: tuple[int | None, int | None], interpolation: str | int | INTER = INTER.BILINEAR, return_scale: bool = False)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **說明**：對輸入影像進行縮放處理。

- **參數**

  - **img** (`np.ndarray`)：要進行縮放處理的輸入影像。
  - **size** (`tuple[int | None, int | None]`)：縮放後的影像大小（格式為 `(height, width)`）。若其中一個維度為 `None`，會維持長寬比自動推算另一邊。
  - **interpolation** (`str | int | INTER`)：插值方法。可用選項有：`INTER.NEAREST`、`INTER.BILINEAR`、`INTER.CUBIC`、`INTER.AREA`、`INTER.LANCZOS4`。預設為 `INTER.BILINEAR`。
  - **return_scale** (`bool`)：是否返回縮放比例。預設為 False。

- **傳回值**

  - **np.ndarray**：縮放後的影像。
  - **Tuple[np.ndarray, float, float]**：縮放後的影像和寬高縮放比例。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')

  # Resize the image to H=256, W=256
  resized_img = cb.imresize(img, [256, 256])

  # Resize the image to H=256, keep the aspect ratio
  resized_img = cb.imresize(img, [256, None])

  # Resize the image to W=256, keep the aspect ratio
  resized_img = cb.imresize(img, [None, 256])
  ```
