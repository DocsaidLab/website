# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: _Polygon, color: _Color = (0, 255, 0), thickness: _Thickness = 2, fillup: bool = False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依賴**

  - 請先安裝 `capybara-docsaid[visualization]`。

- **說明**：在影像上繪製多邊形。

- **參數**

  - **img** (`np.ndarray`)：要繪製的影像，為 NumPy 陣列。
  - **polygon** (`Union[Polygon, np.ndarray]`)：要繪製的多邊形，可以是多邊形物件或 NumPy 陣列形式的 [[x1, y1], [x2, y2], ...]。
  - **color** (`_Color`)：要繪製的多邊形顏色（BGR）。預設為 (0, 255, 0)。
  - **thickness** (`_Thickness`)：要繪製的多邊形邊線的粗細。預設為 2。
  - **fillup** (`bool`)：是否填滿多邊形。預設為 False。
  - **kwargs**：其他參數。

- **傳回值**

  - **np.ndarray**：繪製了多邊形的影像。

- **範例**

  ```python
  from capybara import Polygon, imread
  from capybara.vision.visualization.draw import draw_polygon

  img = imread('lena.png')
  polygon = Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  polygon_img = draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
  ```

  ![draw_polygon](./resource/test_draw_polygon.jpg)
