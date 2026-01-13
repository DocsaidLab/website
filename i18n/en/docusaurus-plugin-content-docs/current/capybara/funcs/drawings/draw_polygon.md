# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: _Polygon, color: _Color = (0, 255, 0), thickness: _Thickness = 2, fillup: bool = False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **Dependencies**

  - Install `capybara-docsaid[visualization]` first.

- **Description**: Draws a polygon on an image.

- **Parameters**

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **polygon** (`Union[Polygon, np.ndarray]`): The polygon to draw, either as a `Polygon` object or a NumPy array in the format [[x1, y1], [x2, y2], ...].
  - **color** (`_Color`): Polygon color (BGR). Default is (0, 255, 0).
  - **thickness** (`_Thickness`): Line thickness. Default is 2.
  - **fillup** (`bool`): Whether to fill the polygon. Defaults to False.
  - **kwargs**: Other parameters.

- **Returns**

  - **np.ndarray**: The image with the polygon drawn on it.

- **Example**

  ```python
  from capybara import Polygon, imread
  from capybara.vision.visualization.draw import draw_polygon

  img = imread('lena.png')
  polygon = Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  polygon_img = draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
  ```

  ![draw_polygon](./resource/test_draw_polygon.jpg)
