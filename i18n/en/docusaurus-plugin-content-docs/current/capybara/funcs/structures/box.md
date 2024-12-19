---
sidebar_position: 2
---

# Box

>[Box(array: _Box, box_mode: _BoxMode = BoxMode.XYXY, normalized: bool = False) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/structures/boxes.py#L106)

- **Description**

    `Box` is a class designed to represent bounding boxes. It offers numerous methods for manipulating bounding box coordinates such as converting coordinate systems, normalizing coordinates, denormalizing coordinates, cropping, moving, and scaling bounding boxes, among others.

- **Parameters**

    - **array** (`_Box`): A bounding box.
    - **box_mode** (`_BoxMode`): An enumeration class that represents the different ways of defining a bounding box, with the default format being `XYXY`.
    - **normalized** (`bool`): Indicates whether the bounding box coordinates are normalized. Default is `False`.

- **Attributes**

    - **box_mode**: Retrieves the representation mode of the bounding box.
    - **normalized**: Retrieves the normalization status of the bounding box.
    - **width**: Retrieves the width of the bounding box.
    - **height**: Retrieves the height of the bounding box.
    - **left_top**: Retrieves the top-left corner of the bounding box.
    - **right_bottom**: Retrieves the bottom-right corner of the bounding box.
    - **left_bottom**: Retrieves the bottom-left corner of the bounding box.
    - **right_top**: Retrieves the top-right corner of the bounding box.
    - **area**: Retrieves the area of the bounding box.
    - **aspect_ratio**: Calculates the aspect ratio of the bounding box.
    - **center**: Calculates the center point of the bounding box.

- **Methods**

    - **convert**(`to_mode: _BoxMode`): Converts the format of the bounding box.
    - **copy**(): Copies the bounding box object.
    - **numpy**(): Converts the bounding box object to a numpy array.
    - **square**(): Transforms the bounding box into a square bounding box.
    - **normalize**(`w: int, h: int`): Normalizes the coordinates of the bounding box.
    - **denormalize**(`w: int, h: int`): Denormalizes the coordinates of the bounding box.
    - **clip**(`xmin: int, ymin: int, xmax: int, ymax: int`): Crops the bounding box.
    - **shift**(`shift_x: float, shift_y: float`): Moves the bounding box.
    - **scale**(`dsize: Tuple[int, int] = None, fx: float = None, fy: float = None`): Scales the bounding box.
    - **to_list**(): Converts the bounding box to a list.
    - **to_polygon**(): Converts the bounding box to a polygon (docsaidkit.Polygon).

- **Example**

    ```python
    import docsaidkit as D

    box = D.Box([10, 20, 50, 80])
    print(box)
    # >>> Box([10. 20. 50. 80.]), BoxMode.XYXY

    box1 = box.convert(D.BoxMode.XYWH)
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
