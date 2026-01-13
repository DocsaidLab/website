# imwarp_quadrangles

> [imwarp_quadrangles(img: np.ndarray, polygons: Polygons, dst_size: tuple[int, int] | None = None, do_order_points: bool = True) -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **說明**：對輸入影像應用給定「多個」多邊形定義的 4 點透視變換。函數為會自動對四個點進行排序，其順序：第一個點為左上角，第二個點為右上角，第三個點為右下角，第四個點為左下角。影像變換的目標大小的長寬由多邊形的最小旋轉外接矩形的長寬決定。

- **參數**

  - **img** (`np.ndarray`)：要進行變換的輸入影像。
  - **polygons** (`Polygons`)：包含多個四點多邊形的 `Polygons` 物件。
  - **dst_size** (`tuple[int, int] | None`)：輸出影像尺寸（格式為 `(width, height)`）。若為 `None`，則由 `polygon.min_box_wh` 推算。
  - **do_order_points** (`bool`)：是否將四點排序為順時針順序（左上、右上、右下、左下）。預設為 `True`。

- **傳回值**

  - **List[np.ndarray]**：變換後的影像列表。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('./resource/test_warp.jpg')
  polygons = cb.Polygons([[[602, 404], [1832, 530], [1588, 985], [356, 860]]])
  warp_imgs = cb.imwarp_quadrangles(img, polygons)
  ```

  圖示請參考 [**imwarp_quadrangle**](./imwarp_quadrangle.md)。
