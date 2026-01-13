# draw_box

> [draw_box(img: np.ndarray, box: _Box, color: _Color = (0, 255, 0), thickness: _Thickness = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **Dependencies**

  - Drawing utilities are optional. Install first:

    ```bash
    pip install "capybara-docsaid[visualization]"
    ```

- **Description**: Draws a bounding box on an image.

- **Parameters**

  - **img** (`np.ndarray`): The image to draw on, as a NumPy array.
  - **box** (`Union[Box, np.ndarray]`): The bounding box to draw, either as a `Box` object or a NumPy array in the format [x1, y1, x2, y2].
  - **color** (`_Color`): Box color (OpenCV convention: BGR). Default is (0, 255, 0).
  - **thickness** (`_Thickness`): Line thickness. Default is 2.

- **Returns**

  - **np.ndarray**: The image with the bounding box drawn.

- **Example**

  ```python
  from capybara import Box, imread
  from capybara.vision.visualization.draw import draw_box

  img = imread('lena.png')
  box = Box([20, 20, 100, 100])
  box_img = draw_box(img, box, color=(0, 255, 0), thickness=2)
  ```

  ![draw_box](./resource/test_draw_box.jpg)
