---
sidebar_position: 3
---

# Boxes

>[Boxes(array: _Boxes, box_mode: _BoxMode = BoxMode.XYXY, normalized: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/boxes.py#L361)

- **Description**

    `Boxes` is a class designed to represent multiple bounding boxes. It offers numerous methods for manipulating the coordinates of multiple bounding boxes, such as converting coordinate systems, normalizing coordinates, denormalizing coordinates, cropping bounding boxes, moving bounding boxes, and scaling bounding boxes.

- **Parameters**

    - **array** (`_Boxes`): A collection of bounding boxes.
    - **box_mode** (`_BoxMode`): An enumeration class that represents the different ways of defining bounding boxes, with the default format being `XYXY`.
    - **normalized** (`bool`): Indicates whether the bounding box coordinates are normalized. Default is `False`.

- **Attributes**

    - **box_mode**: Retrieves the representation mode of the bounding boxes.
    - **normalized**: Retrieves the normalization status of the bounding boxes.
    - **width**: Retrieves the width of the bounding boxes.
    - **height**: Retrieves the height of the bounding boxes.
    - **left_top**: Retrieves the top-left corner of the bounding boxes.
    - **right_bottom**: Retrieves the bottom-right corner of the bounding boxes.
    - **area**: Retrieves the area of the bounding boxes.
    - **aspect_ratio**: Calculates the aspect ratio of the bounding boxes.
    - **center**: Calculates the center point of the bounding boxes.

- **Methods**

    - **convert**(`to_mode: _BoxMode`): Converts the format of the bounding boxes.
    - **copy**(): Copies the bounding boxes object.
    - **numpy**(): Converts the bounding boxes object to a numpy array.
    - **square**(): Transforms the bounding boxes into square bounding boxes.
    - **normalize**(`w: int, h: int`): Normalizes the coordinates of the bounding boxes.
    - **denormalize**(`w: int, h: int`): Denormalizes the coordinates of the bounding boxes.
    - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`): Crops the bounding boxes.
    - **shift**(`shift_x: float, shift_y: float`): Moves the bounding boxes.
    - **scale**(`dsize: Tuple[int, int] = None, fx: float = None, fy: float = None`): Scales the bounding boxes.
    - **to_list**(): Converts the bounding boxes to a list.
    - **to_polygons**(): Converts the bounding boxes to polygons (docsaidkit.Polygons).

- **Example**

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
    - **to_polygons**()：將邊界框轉換為多邊形(docsaidkit.Polygons)。

- **範例**

    ```python
    import docsaidkit as D

    boxes = D.Boxes([[10, 20, 50, 80], [20, 30, 60, 90]])
    print(boxes)
    # >>> Boxes([[10. 20. 50. 80.], [20. 30. 60. 90.]]), BoxMode.XYXY

    boxes1 = boxes.convert(D.BoxMode.XYWH)
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
