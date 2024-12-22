# imresize

> [imresize(img: np.ndarray, size: Tuple[int, int], interpolation: Union[str, int, INTER] = INTER.BILINEAR, return_scale: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L15)

- **說明**：對輸入影像進行縮放處理。

- **參數**

  - **img** (`np.ndarray`)：要進行縮放處理的輸入影像。
  - **size** (`Tuple[int, int]`)：縮放後的影像大小。如果只給定一個維度，則保持原始影像的寬高比計算另一個維度。
  - **interpolation** (`Union[str, int, INTER]`)：插值方法。可用選項有：INTER.NEAREST, INTER.LINEAR, INTER.CUBIC, INTER.LANCZOS4。預設為 INTER.LINEAR。
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
