---
sidebar_position: 4
---

# Polygon

> [Polygon(array: \_Polygon, normalized: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/polygons.py#L64)

- **Description**:

  `Polygon` is a class used to represent polygons. This class provides several methods for manipulating polygon coordinates, such as normalizing coordinates, denormalizing coordinates, clipping polygons, shifting polygons, scaling polygons, converting polygons to convex hulls, converting polygons to minimum bounding rectangles, converting polygons to bounding boxes, etc.

- **Parameters**

  - **array** (`_Polygon`): The coordinates of the polygon.
  - **normalized** (`bool`): A flag indicating whether the polygon coordinates are normalized. Default is `False`.

- **Attributes**

  - **normalized**: Gets the normalized status of the polygon.
  - **moments**: Gets the moments of the polygon.
  - **area**: Gets the area of the polygon.
  - **arclength**: Gets the perimeter (arc length) of the polygon.
  - **centroid**: Gets the centroid of the polygon.
  - **boundingbox**: Gets the bounding box of the polygon.
  - **min_circle**: Gets the minimum enclosing circle of the polygon.
  - **min_box**: Gets the minimum bounding rectangle of the polygon.
  - **orientation**: Gets the orientation of the polygon.
  - **min_box_wh**: Gets the width and height of the polygon's minimum bounding rectangle.
  - **extent**: Gets the extent (occupancy ratio) of the polygon.
  - **solidity**: Gets the solidity of the polygon.

- **Methods**

  - **copy**(): Copies the polygon object.
  - **numpy**(): Converts the polygon object to a numpy array.
  - **normalize**(`w: float, h: float`): Normalizes the polygon coordinates.
  - **denormalize**(`w: float, h: float`): Denormalizes the polygon coordinates.
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`): Clips the polygon.
  - **shift**(`shift_x: float, shift_y: float`): Shifts the polygon.
  - **scale**(`distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre`): Scales the polygon.
  - **to_convexhull**(): Converts the polygon to a convex hull.
  - **to_min_boxpoints**(): Converts the polygon to the coordinates of the minimum bounding rectangle.
  - **to_box**(`box_mode: str = 'xyxy'`): Converts the polygon to a bounding box.
  - **to_list**(`flatten: bool = False`): Converts the polygon to a list.
  - **is_empty**(`threshold: int = 3`): Determines if the polygon is empty.

- **Example**

  ```python
  import capybara as cb

  polygon = cb.Polygon([[10., 20.], [50, 20.], [50, 80.], [10., 80.]])
  print(polygon)
  # >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])

  polygon1 = polygon.normalize(100, 100)
  print(polygon1)
  # >>> Polygon([[0.1 0.2], [0.5 0.2], [0.5 0.8], [0.1 0.8]])

  polygon2 = polygon.denormalize(100, 100)
  print(polygon2)
  # >>> Polygon([[1000. 2000.], [5000. 2000.], [5000. 8000.], [1000. 8000.]])

  polygon3 = polygon.clip(20, 20, 60, 60)
  print(polygon3)
  # >>> Polygon([[20. 20.], [50. 20.], [50. 60.], [20. 60.]])

  polygon4 = polygon.shift(10, 10)
  print(polygon4)
  # >>> Polygon([[20. 30.], [60. 30.], [60. 90.], [20. 90.]])

  polygon5 = polygon.scale(10)
  print(polygon5)
  # >>> Polygon([[0. 10.], [60. 10.], [60. 90.], [0. 90.]])

  polygon6 = polygon.to_convexhull()
  print(polygon6)
  # >>> Polygon([[50. 80.], [10. 80.], [10. 20.], [50. 20.]])

  polygon7 = polygon.to_min_boxpoints()
  print(polygon7)
  # >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])

  polygon8 = polygon.to_box('xywh')
  print(polygon8)
  # >>> Box([10. 20. 40. 60.]), BoxMode.XYWH

  polygon9 = polygon.to_list()
  print(polygon9)
  # >>> [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]]

  polygon10 = polygon.is_empty()
  print(polygon10)
  # >>> False
  ```
