# draw_polygons

> [draw_polygons(img: np.ndarray, polygons: _Polygons, colors: _Colors = (0, 255, 0), thicknesses: _Thicknesses = 2, fillup: bool = False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依賴**

  - 請先安裝 `capybara-docsaid[visualization]`。

- **說明**：在影像上繪製多個多邊形。

- **參數**

  - **img** (`np.ndarray`)：要繪製的影像，為 NumPy 陣列。
  - **polygons** (`List[Union[Polygon, np.ndarray]]`)：要繪製的多邊形，可以是多邊形物件的列表或 NumPy 陣列形式的 [[x1, y1], [x2, y2], ...]。
  - **colors** (`_Colors`)：要繪製的多邊形顏色（BGR）。可以是單一顏色或顏色列表。預設為 (0, 255, 0)。
  - **thicknesses** (`_Thicknesses`)：要繪製的多邊形邊線粗細。可以是單一值或列表。預設為 2。
  - **fillup** (`bool`)：是否填滿多邊形。預設為 False。
  - **kwargs**：其他參數。

- **傳回值**

  - **np.ndarray**：繪製了多邊形的影像。

- **範例**

  ```python
  from capybara import Polygon, imread
  from capybara.vision.visualization.draw import draw_polygons

  img = imread('lena.png')
  polygons = [
      Polygon([(20, 20), (100, 20), (80, 80), (20, 40)]),
      Polygon([(100, 100), (20, 100), (40, 40), (100, 80)])
  ]
  polygons_img = draw_polygons(
      img,
      polygons,
      colors=[(0, 255, 0), (255, 0, 0)],
      thicknesses=2,
  )
  ```

  ![draw_polygons](./resource/test_draw_polygons.jpg)
