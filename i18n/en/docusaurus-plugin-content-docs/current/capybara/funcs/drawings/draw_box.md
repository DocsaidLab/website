# draw_box

> [draw_box(img: np.ndarray, box: Union[Box, np.ndarray], color: \_Color = (0, 255, 0), thickness: \_Thickness = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L36)

- **Description**: Draws a bounding box on an image.

- **Parameters**:

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **box** (`Union[Box, np.ndarray]`): The bounding box to draw, either as a `Box` object or a NumPy array in the format [x1, y1, x2, y2].
  - **color** (`_Color`): The color of the bounding box. Defaults to (0, 255, 0) (green).
  - **thickness** (`_Thickness`): The thickness of the box border. Defaults to 2.

- **Return Value**:

  - **np.ndarray**: The image with the bounding box drawn.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  box = cb.Box([20, 20, 100, 100])
  box_img = cb.draw_box(img, box, color=(0, 255, 0), thickness=2)
  ```

  ![draw_box](./resource/test_draw_box.jpg)
