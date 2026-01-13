# draw_text

> [draw_text(img: np.ndarray, text: str, location: _Point, color: _Color = (0, 0, 0), text_size: int = 12, font_path: str | Path | None = None, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **Dependencies**

  - Install `capybara-docsaid[visualization]` first.

- **Description**: Draws text at a specified location on an image.

- **Parameters**

  - **img** (`np.ndarray`): The image to draw on.
  - **text** (`str`): The text to draw.
  - **location** (`_Point`): The (x, y) location.
  - **color** (`_Color`): Text color (BGR). Default is (0, 0, 0).
  - **text_size** (`int`): The size of the text. Defaults to 12.
  - **font_path** (`str | Path | None`): Path to a font file. If not provided, it falls back to the built-in font (`NotoSansMonoCJKtc-VF.ttf`) and then `PIL.ImageFont.load_default()`.
  - **kwargs**: Additional parameters for drawing, depending on the underlying library or method used.

- **Returns**

  - **np.ndarray**: The image with the text drawn on it.

- **Example**

  ```python
  from capybara import imread
  from capybara.vision.visualization.draw import draw_text

  img = imread('lena.png')
  text_img = draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
  ```

  ![draw_text](./resource/test_draw_text.jpg)
