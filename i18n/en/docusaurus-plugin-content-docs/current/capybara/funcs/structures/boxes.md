---
sidebar_position: 3
---

# Boxes

> [Boxes(array: \_Boxes, box_mode: \_BoxMode = BoxMode.XYXY, normalized: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L362)

- **Description**:

  `Boxes` is a class used to represent multiple bounding boxes. This class provides several methods to manipulate the coordinates of multiple bounding boxes, such as converting coordinate systems, normalizing coordinates, denormalizing coordinates, clipping bounding boxes, shifting bounding boxes, scaling bounding boxes, etc.

- **Parameters**

  - **array** (`_Boxes`): Multiple bounding boxes.
  - **box_mode** (`_BoxMode`): An enumeration class representing different bounding box formats. The default format is `XYXY`.
  - **normalized** (`bool`): A flag indicating whether the bounding box coordinates are normalized. Default is `False`.

- **Attributes**

  - **box_mode**: Gets the representation of the bounding box format.
  - **normalized**: Gets the normalized status of the bounding box.
  - **width**: Gets the width of the bounding box.
  - **height**: Gets the height of the bounding box.
  - **left_top**: Gets the top-left corner of the bounding box.
  - **right_bottom**: Gets the bottom-right corner of the bounding box.
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
  - **to_polygons**(): Converts the bounding box to polygons (capybara.Polygons).

- **Example**

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
