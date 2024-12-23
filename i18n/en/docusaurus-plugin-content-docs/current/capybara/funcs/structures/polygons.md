---
sidebar_position: 5
---

# Polygons

> [Polygons(array: \_Polygons, normalized: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/polygons.py#L336)

- **Description**:

  `Polygons` is a class used to represent multiple polygons. This class provides several methods for manipulating the coordinates of multiple polygons, such as normalizing coordinates, denormalizing coordinates, clipping polygons, shifting polygons, scaling polygons, converting polygons to convex hulls, converting polygons to minimum bounding rectangles, converting polygons to bounding boxes, etc.

- **Parameters**

  - **array** (`_Polygons`): The coordinates of multiple polygons.
  - **normalized** (`bool`): A flag indicating whether the polygon coordinates are normalized. Default is `False`.

- **Attributes**

  - **normalized**: Gets the normalized status of the polygons.
  - **moments**: Gets the moments of the polygons.
  - **area**: Gets the area of the polygons.
  - **arclength**: Gets the perimeter (arc length) of the polygons.
  - **centroid**: Gets the centroid of the polygons.
  - **boundingbox**: Gets the bounding box of the polygons.
  - **min_circle**: Gets the minimum enclosing circle of the polygons.
  - **min_box**: Gets the minimum bounding rectangle of the polygons.
  - **orientation**: Gets the orientation of the polygons.
  - **min_box_wh**: Gets the width and height of the polygons' minimum bounding rectangles.
  - **extent**: Gets the extent (occupancy ratio) of the polygons.
  - **solidity**: Gets the solidity of the polygons.

- **Methods**

  - **copy**(): Copies the polygon object.
  - **numpy**(): Converts the polygon object to a numpy array.
  - **normalize**(`w: float, h: float`): Normalizes the polygon coordinates.
  - **denormalize**(`w: float, h: float`): Denormalizes the polygon coordinates.
  - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`): Clips the polygons.
  - **shift**(`shift_x: float, shift_y: float`): Shifts the polygons.
  - **scale**(`distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre`): Scales the polygons.
  - **to_convexhull**(): Converts the polygons to convex hulls.
  - **to_min_boxpoints**(): Converts the polygons to the coordinates of the minimum bounding rectangle.
  - **to_box**(`box_mode: str = 'xyxy'`): Converts the polygons to bounding boxes.
  - **to_list**(`flatten: bool = False`): Converts the polygons to a list.
  - **is_empty**(`threshold: int = 3`): Determines if the polygons are empty.

- **Class Construction**

  - **from_image**(`image: np.ndarray, mode: int = cv2.RETR_EXTERNAL, method: int = cv2.CHAIN_APPROX_SIMPLE`): Extracts polygon coordinates from an image.
  - **cat**(`polygons_list: List["Polygons"]`): Concatenates multiple lists of polygons into one list of polygons.

- **Example**

  ```python
  import capybara as cb

  polygons = cb.Polygons([
      [[10., 20.], [50, 20.], [50, 80.], [10., 80.]],
      [[60., 20.], [100, 20.], [100, 80.], [60., 80.]]
  ])
  print(polygons)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [10. 20.]
  #           [50. 20.]
  #           [50. 80.]
  #           [10. 80.]
  #       ]),
  #       Polygon([
  #           [60. 20.]
  #           [100. 20.]
  #           [100. 80.]
  #           [60. 80.]
  #       ])
  #   ]
  # )

  polygon1 = polygons.normalize(100, 100)
  print(polygon1)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [0.1 0.2]
  #           [0.5 0.2]
  #           [0.5 0.8]
  #           [0.1 0.8]
  #       ]),
  #       Polygon([
  #           [0.6 0.2]
  #           [1.  0.2]
  #           [1.  0.8]
  #           [0.6 0.8]
  #       ])
  #   ]
  # )

  polygon2 = polygons.denormalize(100, 100)
  print(polygon2)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [1000. 2000.]
  #           [5000. 2000.]
  #           [5000. 8000.]
  #           [1000. 8000.]
  #       ]),
  #       Polygon([
  #           [6000. 2000.]
  #           [10000. 2000.]
  #           [10000. 8000.]
  #           [6000. 8000.]
  #       ])
  #   ]
  # )

  polygon3 = polygons.clip(20, 20, 60, 60)
  print(polygon3)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [20. 20.]
  #           [50. 20.]
  #           [50. 60.]
  #           [20. 60.]
  #       ]),
  #       Polygon([
  #           [60. 20.]
  #           [60. 20.]
  #           [60. 60.]
  #           [60. 60.]
  #       ])
  #   ]
  # )

  polygon4 = polygons.shift(10, 10)
  print(polygon4)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [20. 30.]
  #           [60. 30.]
  #           [60. 90.]
  #           [20. 90.]
  #       ]),
  #       Polygon([
  #           [ 70.  30.]
  #           [110.  30.]
  #           [110.  90.]
  #           [ 70.  90.]
  #       ])
  #   ]
  # )


  polygon5 = polygons.scale(10)
  print(polygon5)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [ 0. 10.]
  #           [60. 10.]
  #           [60. 90.]
  #           [ 0. 90.]
  #       ]),
  #       Polygon([
  #           [ 50.  10.]
  #           [110.  10.]
  #           [110.  90.]
  #           [ 50.  90.]
  #       ])
  #   ]
  # )

  polygon6 = polygons.to_convexhull()
  print(polygon6)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [50. 80.],
  #           [10. 80.],
  #           [10. 20.],
  #           [50. 20.]
  #       ]),
  #       Polygon([
  #           [100.  80.],
  #           [ 60.  80.],
  #           [ 60.  20.],
  #           [100.  20.]
  #       ])
  #   ]
  # )

  polygon7 = polygons.to_min_boxpoints()
  print(polygon7)
  # >>> Polygons(
  #   [
  #       Polygon([
  #           [10. 20.]
  #           [50. 20.]
  #           [50. 80.]
  #           [10. 80.]
  #       ]),
  #       Polygon([
  #           [ 60.  20.]
  #           [100.  20.]
  #           [100.  80.]
  #           [ 60.  80.]
  #       ])
  #   ]
  # )

  polygon8 = polygons.to_boxes() # notice that the method name is different from Polygon
  print(polygon8)
  # >>> Boxes([[ 10.  20.  50.  80.], [ 60.  20. 100.  80.]]), BoxMode.XYXY

  polygon9 = polygons.to_list()
  print(polygon9)
  # >>> [
  #   [
  #       [10.0, 20.0],
  #       [50.0, 20.0],
  #       [50.0, 80.0],
  #       [10.0, 80.0]
  #   ],
  #   [
  #       [60.0, 20.0],
  #       [100.0, 20.0],
  #       [100.0, 80.0],
  #       [60.0, 80.0]
  #   ]
  # ]

  polygon10 = polygons.is_empty()
  print(polygon10)
  # >>> [False, False]
  ```
