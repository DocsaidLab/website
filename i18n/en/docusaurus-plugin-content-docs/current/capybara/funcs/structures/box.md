---
sidebar_position: 2
---

# Box

> [Box(array: \_Box, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L101)

- **Description**:

  `Box` is a class used to represent a bounding box. This class provides several methods to manipulate the coordinates of the bounding box, such as converting coordinate systems, normalizing coordinates, denormalizing coordinates, clipping the bounding box, shifting the bounding box, scaling the bounding box, etc.

- **Parameters**

  - **array** (`_Box`): A bounding box.
  - **box_mode** (`_BoxMode`): An enumeration class representing different bounding box formats. The default format is `XYXY`.
  - **normalized** (`bool`): A flag indicating whether the bounding box coordinates are normalized. Default is `False`.

- **Attributes**

  - **box_mode**: Gets the representation of the bounding box format.
  - **normalized**: Gets the normalized status of the bounding box.
  - **width**: Gets the width of the bounding box.
  - **height**: Gets the height of the bounding box.
  - **left_top**: Gets the top-left corner of the bounding box.
  - **right_bottom**: Gets the bottom-right corner of the bounding box.
  - **left_bottom**: Gets the bottom-left corner of the bounding box.
  - **right_top**: Gets the top-right corner of the bounding box.
  - **area**: Gets the area of the bounding box.
  - **aspect_ratio**: Calculates the aspect ratio of the bounding box.
  - **center**: Calculates the center of the bounding box.

- **Methods**

  - **convert**(`to_mode: _BoxMode`): Converts the bounding box format.
  - **copy**(): Copies the bounding box object.
  - **numpy**(): Converts the bounding box object to a numpy array.
  - **square**(): Converts the bounding box to a square.
  - **normalize**(`w: int, h: int`): Normalizes the bounding box coordinates.
  - **denormalize**(`w: int, h: int`): Denormalizes the bounding box coordinates.
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`): Clips the bounding box.
  - **shift**(`shift_x: float, shift_y: float`): Shifts the bounding box.
  - **scale**(`dsize: Tuple[int, int] = None, fx: float = None, fy: float = None`): Scales the bounding box.
  - **to_list**(): Converts the bounding box to a list.
  - **to_polygon**(): Converts the bounding box to a polygon (capybara.Polygon).

- **Example**

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
