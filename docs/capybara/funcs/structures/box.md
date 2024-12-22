---
sidebar_position: 2
---

# Box

> [Box(array: \_Box, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L101)

- **說明**：

  `Box` 是一個用來表示邊界框的類別。這個類別提供了許多方法，用來操作邊界框的座標，例如：轉換座標系統、正規化座標、反正規化座標、裁剪邊界框、移動邊界框、縮放邊界框等等。

- **參數**

  - **array** (`_Box`)：一個邊界框。
  - **box_mode** (`_BoxMode`)：表示邊界框的不同方式的列舉類別，預設格式為 `XYXY`。
  - **normalized** (`bool`)：是否為正規化邊界框的座標，是一個屬性標記。預設為 `False`。

- **屬性**

  - **box_mode**：取得邊界框的表示方式。
  - **normalized**：取得邊界框的正規化狀態。
  - **width**：取得邊界框的寬度。
  - **height**：取得邊界框的高度。
  - **left_top**：取得邊界框的左上角點。
  - **right_bottom**：取得邊界框的右下角點。
  - **left_bottom**：取得邊界框的左下角點。
  - **right_top**：取得邊界框的右上角點。
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
  - **to_polygon**()：將邊界框轉換為多邊形(capybara.Polygon)。

- **範例**

  ```python
  import capybara as cb

  box = cb.Box([10, 20, 50, 80])
  print(box)
  # >>> Box([10. 20. 50. 80.]), BoxMode.XYXY

  box1 = box.convert(cb.BoxMode.XYWH)
  print(box1)
  # >>> Box([10. 20. 40. 60.]), BoxMode.XYWH

  box2 = box.normalize(100, 100)
  print(box2)
  # >>> Box([0.1 0.2 0.5 0.8]), BoxMode.XYXY

  box3 = box.denormalize(100, 100)
  print(box3)
  # >>> Box([1000. 2000. 5000. 8000.]), BoxMode.XYXY

  box4 = box.clip(0, 0, 50, 50)
  print(box4)
  # >>> Box([10. 20. 50. 50.]), BoxMode.XYXY

  box5 = box.shift(10, 10)
  print(box5)
  # >>> Box([20. 30. 60. 90.]), BoxMode.XYXY

  box6 = box.scale(dsize=(10, 10))
  print(box6)
  # >>> Box([5. 15. 55. 85.]), BoxMode.XYXY

  box7 = box.scale(fx=1.1, fy=1.1)
  print(box7)
  # >>> Box([8. 17. 52. 83.]), BoxMode.XYXY

  box8 = box.square()
  print(box8)
  # >>> Box([10. 30. 50. 70.]), BoxMode.XYXY

  box9 = box.to_list()
  print(box9)
  # >>> [10.0, 20.0, 50.0, 80.0]

  poly = box.to_polygon()
  print(poly)
  # >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])
  ```
