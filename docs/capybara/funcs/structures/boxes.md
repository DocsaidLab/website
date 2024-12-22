---
sidebar_position: 3
---

# Boxes

> [Boxes(array: \_Boxes, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L362)

- **說明**：

  `Boxes` 是一個用來表示多個邊界框的類別。這個類別提供了許多方法，用來操作多個邊界框的座標，例如：轉換座標系統、正規化座標、反正規化座標、裁剪邊界框、移動邊界框、縮放邊界框等等。

- **參數**

  - **array** (`_Boxes`)：多個邊界框。
  - **box_mode** (`_BoxMode`)：表示邊界框的不同方式的列舉類別，預設格式為 `XYXY`。
  - **normalized** (`bool`)：是否為正規化邊界框的座標，是一個屬性標記。預設為 `False`。

- **屬性**

  - **box_mode**：取得邊界框的表示方式。
  - **normalized**：取得邊界框的正規化狀態。
  - **width**：取得邊界框的寬度。
  - **height**：取得邊界框的高度。
  - **left_top**：取得邊界框的左上角點。
  - **right_bottom**：取得邊界框的右下角點。
  - **area**：取得邊界框的面積。
  - **aspect_ratio**：計算邊界框的寬高比。
  - **center**：計算邊界框的中心點。

- **方法**

  - **convert**(`to_mode: _BoxMode`)：轉換邊界框的格式。
  - **copy**()：複製邊界框物件。
  - **numpy**()：將邊界框物件轉換為 numpy 陣列。
  - **square**()：將邊界框轉換為正方形邊界框。
  - **normalize**(`w: int, h: int`)：正規化邊界框的座標。
  - **denormalize**(`w: int, h: int`)：反正規化邊界框的座標。
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`)：裁剪邊界框。
  - **shift**(`shift_x: float, shift_y: float`)：移動邊界框。
  - **scale**(`dsize: Tuple[int, int] = None, fx: float = None, fy: float = None`)：縮放邊界框。
  - **to_list**()：將邊界框轉換為列表。
  - **to_polygons**()：將邊界框轉換為多邊形(capybara.Polygons)。

- **範例**

  ```python
  import capybara as cb

  boxes = cb.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
  print(boxes)
  # >>> Boxes([[10. 20. 50. 80.], [20. 30. 60. 90.]]), BoxMode.XYXY

  boxes1 = boxes.convert(cb.BoxMode.XYWH)
  print(boxes1)
  # >>> Boxes([[10. 20. 40. 60.], [20. 30. 40. 60.]]), BoxMode.XYWH

  boxes2 = boxes.normalize(100, 100)
  print(boxes2)
  # >>> Boxes([[0.1 0.2 0.5 0.8], [0.2 0.3 0.6 0.9]]), BoxMode.XYXY

  boxes3 = boxes.denormalize(100, 100)
  print(boxes3)
  # >>> Boxes([[1000. 2000. 5000. 8000.], [2000. 3000. 6000. 9000.]]), BoxMode.XYXY

  boxes4 = boxes.clip(0, 0, 50, 50)
  print(boxes4)
  # >>> Boxes([[10. 20. 50. 50.], [20. 30. 50. 50.]]), BoxMode.XYXY

  boxes5 = boxes.shift(10, 10)
  print(boxes5)
  # >>> Boxes([[20. 30. 60. 90.], [30. 40. 70. 100.]]), BoxMode.XYXY

  boxes6 = boxes.scale(dsize=(10, 10))
  print(boxes6)
  # >>> Boxes([[5. 15. 55. 85.], [15. 25. 65. 95.]]), BoxMode.XYXY

  boxes7 = boxes.square()
  print(boxes7)
  # >>> Boxes([[0. 20. 60. 80.], [10. 30. 70. 90.]]), BoxMode.XYXY

  boxes8 = boxes.to_list()
  print(boxes8)
  # >>> [[10.0, 20.0, 50.0, 80.0], [20.0, 30.0, 60.0, 90.0]]

  polys = boxes.to_polygons() # Notice: It's different from Box.to_polygon()
  print(polys)
  # >>> Polygons([
  #       Polygon([
  #           [10. 20.]
  #           [50. 20.]
  #           [50. 80.]
  #           [10. 80.]
  #       ]),
  #       Polygon([
  #           [20. 30.]
  #           [60. 30.]
  #           [60. 90.]
  #           [20. 90.]
  #       ])
  #    ])
  ```
