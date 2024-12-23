# draw_text

> [draw_text(img: np.ndarray, text: str, location: np.ndarray, color: tuple = (0, 0, 0), text_size: int = 12, font_path: str = None, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L197)

- **Description**: Draws text at a specified location on an image.

- **Parameters**:

  - **img** (`np.ndarray`): The image to draw on.
  - **text** (`str`): The text to draw.
  - **location** (`np.ndarray`): The x, y coordinates where the text should be drawn.
  - **color** (`tuple`): The RGB color of the text. Defaults to black (0, 0, 0).
  - **text_size** (`int`): The size of the text. Defaults to 12.
  - **font_path** (`str`): The path to the font file to use. If not provided, the default font "NotoSansMonoCJKtc-VF.ttf" will be used.
  - **kwargs**: Additional parameters for drawing, depending on the underlying library or method used.

- **Return Value**:

  - **np.ndarray**: The image with the text drawn on it.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  text_img = cb.draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
  ```

  ![draw_text](./resource/test_draw_text.jpg)
