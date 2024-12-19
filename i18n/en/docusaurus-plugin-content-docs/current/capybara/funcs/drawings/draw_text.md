---
sidebar_position: 5
---

# draw_text

> [draw_text(img: np.ndarray, text: str, location: np.ndarray, color: tuple = (0, 0, 0), text_size: int = 12, font_path: str = None, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L211)

- **Description**

    Draw text at a specified position on an image.

- **Parameters**

    - **img** (`np.ndarray`): The image to draw on.
    - **text** (`str`): The text to draw.
    - **location** (`np.ndarray`): The x, y coordinates where the text should be drawn.
    - **color** (`tuple`): The RGB value of the text color. Defaults to black (0, 0, 0).
    - **text_size** (`int`): The size of the text to draw. Defaults to 12.
    - **font_path** (`str`): The path to the font file to use. If not provided, the default font "NotoSansMonoCJKtc-VF.ttf" is used.
    - **kwargs**: Additional parameters for drawing, depending on the underlying library or method used.

- **Returns**

    - **np.ndarray**: The image with the drawn text.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    text_img = D.draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
    ```

    ![draw_text](./resource/test_draw_text.jpg)
