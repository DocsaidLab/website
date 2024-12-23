# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: Union[Polygon, np.ndarray], color: \_Color = (0, 255, 0), thickness: \_Thickness = 2, fillup=False, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L103)

- **Description**: Draws a polygon on an image.

- **Parameters**:

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **polygon** (`Union[Polygon, np.ndarray]`): The polygon to draw, either as a `Polygon` object or a NumPy array in the format [[x1, y1], [x2, y2], ...].
  - **color** (`_Color`): The color of the polygon. Defaults to (0, 255, 0) (green).
  - **thickness** (`_Thickness`): The thickness of the polygon's border. Defaults to 2.
  - **fillup** (`bool`): Whether to fill the polygon. Defaults to False.
  - **kwargs**: Other parameters.

- **Return Value**:

  - **np.ndarray**: The image with the polygon drawn on it.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  polygon = cb.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  polygon_img = cb.draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
  ```

  ![draw_polygon](./resource/test_draw_polygon.jpg)
