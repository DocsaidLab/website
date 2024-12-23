# draw_polygons

> [draw_polygons(img: np.ndarray, polygons: Polygons, color: \_Colors = (0, 255, 0), thickness: \_Thicknesses = 2, fillup=False, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L150)

- **Description**: Draws multiple polygons on an image.

- **Parameters**:

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **polygons** (`List[Union[Polygon, np.ndarray]]`): The polygons to draw, either as a list of `Polygon` objects or NumPy arrays in the format [[x1, y1], [x2, y2], ...].
  - **color** (`_Colors`): The color of the polygons. It can be a single color or a list of colors. Defaults to (0, 255, 0) (green).
  - **thickness** (`_Thicknesses`): The thickness of the polygon's borders. It can be a single thickness or a list of thicknesses. Defaults to 2.
  - **fillup** (`bool`): Whether to fill the polygons. Defaults to False.
  - **kwargs**: Other parameters.

- **Return Value**:

  - **np.ndarray**: The image with the polygons drawn on it.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  polygons = [
      cb.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)]),
      cb.Polygon([(100, 100), (20, 100), (40, 40), (100, 80)])
  ]
  polygons_img = cb.draw_polygons(img, polygons, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
  ```

  ![draw_polygons](./resource/test_draw_polygons.jpg)
