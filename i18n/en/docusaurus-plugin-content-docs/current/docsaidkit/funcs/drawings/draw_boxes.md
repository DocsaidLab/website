---
sidebar_position: 2
---

# draw_boxes

> [draw_boxes(img: np.ndarray, boxes: Union[Boxes, np.ndarray], color: _Colors = (0, 255, 0), thickness: _Thicknesses = 2) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L67)

- **Description**

    Draw multiple Bounding Boxes on an image.

- **Parameters**

    - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
    - **boxes** (`Union[Boxes, np.ndarray]`): The Bounding Boxes to draw, can be a list of Box objects or a NumPy array in the form [[x1, y1, x2, y2], ...].
    - **color** (`_Colors`): The color(s) of the boxes to draw. Can be a single color or a list of colors. Defaults to (0, 255, 0).
    - **thickness** (`_Thicknesses`): The thickness(es) of the box outlines to draw. Can be a single thickness or a list of thicknesses. Defaults to 2.

- **Returns**

    - **np.ndarray**: The image with the drawn boxes.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    boxes = [D.Box([20, 20, 100, 100]), D.Box([150, 150, 200, 200])]
    boxes_img = D.draw_boxes(img, boxes, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
    ```

    ![draw_boxes](./resource/test_draw_boxes.jpg)
