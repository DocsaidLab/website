# imrotate

> [imrotate(img: np.ndarray, angle: float, scale: float = 1, interpolation: Union[str, int, INTER] = INTER.BILINEAR, bordertype: Union[str, int, BORDER] = BORDER.CONSTANT, bordervalue: Union[int, Tuple[int, int, int]] = None, expand: bool = True, center: Tuple[int, int] = None) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L80)

- **說明**：對輸入影像進行旋轉處理。

- **參數**

  - **img** (`np.ndarray`)：要進行旋轉處理的輸入影像。
  - **angle** (`float`)：旋轉角度。以度為單位，逆時針方向。
  - **scale** (`float`)：縮放比例。預設為 1。
  - **interpolation** (`Union[str, int, INTER]`)：插值方法。可用選項有：INTER.NEAREST, INTER.LINEAR, INTER.CUBIC, INTER.LANCZOS4。預設為 INTER.LINEAR。
  - **bordertype** (`Union[str, int, BORDER]`)：邊界類型。可用選項有：BORDER.CONSTANT, BORDER.REPLICATE, BORDER.REFLECT, BORDER.REFLECT_101。預設為 BORDER.CONSTANT。
  - **bordervalue** (`Union[int, Tuple[int, int, int]]`)：填充邊界的值。僅在 bordertype 為 BORDER.CONSTANT 時有效。預設為 None。
  - **expand** (`bool`)：是否擴展輸出影像以容納整個旋轉後的影像。如果為 True，則擴展輸出影像以使其足夠大以容納整個旋轉後的影像。如果為 False 或省略，則使輸出影像與輸入影像大小相同。請注意，expand 標誌假設圍繞中心旋轉並且沒有平移。預設為 False。
  - **center** (`Tuple[int, int]`)：旋轉中心。預設為影像的中心。

- **傳回值**

  - **np.ndarray**：旋轉後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  rotate_img = cb.imrotate(img, 45, bordertype=D.BORDER.CONSTANT, expand=True)

  # Resize the rotated image to the original size for visualization
  rotate_img = cb.imresize(rotate_img, [img.shape[0], img.shape[1]])
  ```

  ![imrotate](./resource/test_imrotate.jpg)
