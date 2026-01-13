# draw_polygons

> [draw_polygons(img: np.ndarray, polygons: _Polygons, colors: _Colors = (0, 255, 0), thicknesses: _Thicknesses = 2, fillup: bool = False, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **Dependencies**

  - Install `capybara-docsaid[visualization]` first.

- **Description**: Draws multiple polygons on an image.

- **Parameters**

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **polygons** (`List[Union[Polygon, np.ndarray]]`): The polygons to draw, either as a list of `Polygon` objects or NumPy arrays in the format [[x1, y1], [x2, y2], ...].
  - **colors** (`_Colors`): Polygon colors (BGR). A single color or a list. Default is (0, 255, 0).
  - **thicknesses** (`_Thicknesses`): Line thickness(es). A single value or a list. Default is 2.
  - **fillup** (`bool`): Whether to fill the polygons. Defaults to False.
  - **kwargs**: Other parameters.

- **Returns**

  - **np.ndarray**: The image with the polygons drawn on it.

- **Example**

  ```python
  from capybara import Polygon, imread
  from capybara.vision.visualization.draw import draw_polygons

  img = imread('lena.png')
  polygons = [
      Polygon([(20, 20), (100, 20), (80, 80), (20, 40)]),
      Polygon([(100, 100), (20, 100), (40, 40), (100, 80)])
  ]
  polygons_img = draw_polygons(
      img,
      polygons,
      colors=[(0, 255, 0), (255, 0, 0)],
      thicknesses=2,
  )
  ```

  ![draw_polygons](./resource/test_draw_polygons.jpg)
