# draw_boxes

> [draw_boxes(img: np.ndarray, boxes: Union[Boxes, np.ndarray], color: \_Colors = (0, 255, 0), thickness: \_Thicknesses = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L70)

- **Description**: Draws multiple bounding boxes on an image.

- **Parameters**:

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **boxes** (`Union[Boxes, np.ndarray]`): The bounding boxes to draw, which can be a list of `Box` objects or a NumPy array in the format [[x1, y1, x2, y2], ...].
  - **color** (`_Colors`): The color of the bounding boxes. It can be a single color or a list of colors. Defaults to (0, 255, 0) (green).
  - **thickness** (`_Thicknesses`): The thickness of the box borders. It can be a single thickness or a list of thicknesses. Defaults to 2.

- **Return Value**:

  - **np.ndarray**: The image with the bounding boxes drawn.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  boxes = [cb.Box([20, 20, 100, 100]), cb.Box([150, 150, 200, 200])]
  boxes_img = cb.draw_boxes(img, boxes, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
  ```

  ![draw_boxes](./resource/test_draw_boxes.jpg)
