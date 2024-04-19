---
sidebar_position: 4
---

# Polygon

>[Polygon(array: _Polygon, normalized: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/polygons.py#L64)

- **Description**:

    `Polygon` is a class used to represent polygons. This class provides various methods for manipulating polygon coordinates, including normalization, denormalization, clipping, shifting, scaling, converting to convex polygons, converting to minimum bounding rectangles, converting to bounding boxes, and more.

- **Parameters**

    - **array** (`_Polygon`): The coordinates of the polygon.
    - **normalized** (`bool`): Whether the coordinates of the polygon are normalized, a boolean attribute flag. Defaults to `False`.

- **Attributes**

    - **normalized**: Get the normalization status of the polygon.
    - **moments**: Get the moments of the polygon.
    - **area**: Get the area of the polygon.
    - **arclength**: Get the perimeter of the polygon.
    - **centroid**: Get the centroid of the polygon.
    - **boundingbox**: Get the bounding box of the polygon.
    - **min_circle**: Get the minimum enclosing circle of the polygon.
    - **min_box**: Get the minimum bounding rectangle of the polygon.
    - **orientation**: Get the orientation of the polygon.
    - **min_box_wh**: Get the width and height of the minimum bounding rectangle of the polygon.
    - **extent**: Get the extent of the polygon.
    - **solidity**: Get the solidity of the polygon.

- **Methods**

    - **copy**(): Copy the polygon object.
    - **numpy**(): Convert the polygon object to a numpy array.
    - **normalize**(w: float, h: float): Normalize the coordinates of the polygon.
    - **denormalize**(w: float, h: float): Denormalize the coordinates of the polygon.
    - **clip**(xmin: int, ymin: int, xmax: int, ymax: int): Clip the polygon.
    - **shift**(shift_x: float, shift_y: float): Shift the polygon.
    - **scale**(distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre): Scale the polygon.
    - **to_convexhull**(): Convert the polygon to a convex polygon.
    - **to_min_boxpoints**(): Convert the polygon to the coordinates of the minimum bounding rectangle.
    - **to_box**(box_mode: str = 'xyxy'): Convert the polygon to a bounding box.
    - **to_list**(flatten: bool = False): Convert the polygon to a list.
    - **is_empty**(threshold: int = 3): Determine if the polygon is empty.

- **Example**

    ```python
    import docsaidkit as D

    polygon = D.Polygon([[10., 20.], [50, 20.], [50, 80.], [10., 80.]])
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
