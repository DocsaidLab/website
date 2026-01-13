# imwarp_quadrangle

> [imwarp_quadrangle(img: np.ndarray, polygon: Polygon | np.ndarray, dst_size: tuple[int, int] | None = None, do_order_points: bool = True) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **說明**：對輸入影像應用 4 點透視變換。

- **參數**

  - **img** (`np.ndarray`)：要進行變換的輸入影像。
  - **polygon** (`Polygon | np.ndarray`)：包含四個點的多邊形。若為 `np.ndarray`，會先轉成 `Polygon`。
  - **dst_size** (`tuple[int, int] | None`)：輸出影像尺寸（格式為 `(width, height)`）。若為 `None`，則由 `polygon.min_box_wh` 推算。
  - **do_order_points** (`bool`)：是否將四點排序為順時針順序（左上、右上、右下、左下）。預設為 `True`。

- **傳回值**

  - **np.ndarray**：變換後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('./resource/test_warp.jpg')
  polygon = cb.Polygon([[602, 404], [1832, 530], [1588, 985], [356, 860]])
  warp_img = cb.imwarp_quadrangle(img, polygon)
  ```

  ![imwarp_quadrangle](./resource/test_imwarp_quadrangle.jpg)

  其中，上圖的綠框為表示原始多邊形範圍。
