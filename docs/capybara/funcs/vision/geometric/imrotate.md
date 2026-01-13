# imrotate

> [imrotate(img: np.ndarray, angle: float, scale: float = 1, interpolation: str | int | INTER = INTER.BILINEAR, bordertype: str | int | BORDER = BORDER.CONSTANT, bordervalue: int | tuple[int, ...] | None = None, expand: bool = True, center: tuple[int, int] | None = None) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **說明**：對輸入影像進行旋轉處理。

- **參數**

  - **img** (`np.ndarray`)：要進行旋轉處理的輸入影像。
  - **angle** (`float`)：旋轉角度。以度為單位，逆時針方向。
  - **scale** (`float`)：縮放比例。預設為 1。
  - **interpolation** (`str | int | INTER`)：插值方法。可用選項有：`INTER.NEAREST`、`INTER.BILINEAR`、`INTER.CUBIC`、`INTER.AREA`、`INTER.LANCZOS4`。預設為 `INTER.BILINEAR`。
  - **bordertype** (`Union[str, int, BORDER]`)：邊界類型。可用選項有：BORDER.CONSTANT, BORDER.REPLICATE, BORDER.REFLECT, BORDER.REFLECT_101。預設為 BORDER.CONSTANT。
  - **bordervalue** (`int | tuple[int, ...] | None`)：填充邊界的值。僅在 bordertype 為 `BORDER.CONSTANT` 時有效。預設為 `None`（實作中會轉成 0 或 0 tuple）。
  - **expand** (`bool`)：是否擴展輸出影像以容納整個旋轉後的影像。預設為 `True`。
  - **center** (`tuple[int, int] | None`)：旋轉中心；`None` 表示使用影像中心。

- **傳回值**

  - **np.ndarray**：旋轉後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  rotate_img = cb.imrotate(img, 45, bordertype=cb.BORDER.CONSTANT, expand=True)

  # Resize the rotated image to the original size for visualization
  rotate_img = cb.imresize(rotate_img, [img.shape[0], img.shape[1]])
  ```

  ![imrotate](./resource/test_imrotate.jpg)
