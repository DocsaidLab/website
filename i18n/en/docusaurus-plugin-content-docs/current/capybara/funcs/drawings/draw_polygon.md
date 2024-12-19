---
sidebar_position: 3
---

# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: Union[Polygon, np.ndarray], color: _Color = (0, 255, 0), thickness: _Thickness = 2, fillup=False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L106)

- **Description**

    Draw polygons on an image.

- **Parameters**

    - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
    - **polygon** (`Union[Polygon, np.ndarray]`): The polygon(s) to draw, can be a Polygon object or a NumPy array in the form [[x1, y1], [x2, y2], ...].
    - **color** (`_Color`): The color of the polygons to draw. Defaults to (0, 255, 0).
    - **thickness** (`_Thickness`): The thickness of the polygon outlines to draw. Defaults to 2.
    - **fillup** (`bool`): Whether to fill the polygons. Defaults to False.
    - **kwargs**: Additional parameters.

- **Returns**

    - **np.ndarray**: The image with the drawn polygons.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    polygon = D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
    polygon_img = D.draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
    ```

    ![draw_polygon](./resource/test_draw_polygon.jpg)
