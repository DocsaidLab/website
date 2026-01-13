# draw_boxes

> [draw_boxes(img: np.ndarray, boxes: _Boxes, colors: _Colors = (0, 255, 0), thicknesses: _Thicknesses = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依賴**

  - 請先安裝 `capybara-docsaid[visualization]`。

- **說明**：在影像上繪製多個 Bounding Box。

- **參數**

  - **img** (`np.ndarray`)：要繪製的影像，為 NumPy 陣列。
  - **boxes** (`Union[Boxes, np.ndarray]`)：要繪製的 Bounding Box，可以是 Box 物件的列表或 NumPy 陣列形式的 [[x1, y1, x2, y2], ...]。
  - **colors** (`_Colors`)：要繪製的框的顏色（BGR）。可為單一顏色或顏色列表。預設為 (0, 255, 0)。
  - **thicknesses** (`_Thicknesses`)：要繪製的框線粗細。可為單一值或列表。預設為 2。

- **傳回值**

  - **np.ndarray**：繪製了框的影像。

- **範例**

  ```python
  from capybara import Box, imread
  from capybara.vision.visualization.draw import draw_boxes

  img = imread('lena.png')
  boxes = [Box([20, 20, 100, 100]), Box([150, 150, 200, 200])]
  boxes_img = draw_boxes(
      img,
      boxes,
      colors=[(0, 255, 0), (255, 0, 0)],
      thicknesses=2,
  )
  ```

  ![draw_boxes](./resource/test_draw_boxes.jpg)
