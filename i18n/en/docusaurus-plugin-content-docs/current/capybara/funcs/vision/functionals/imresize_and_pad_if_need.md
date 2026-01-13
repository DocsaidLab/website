# imresize_and_pad_if_need

> [imresize_and_pad_if_need(img: np.ndarray, max_h: int, max_w: int, interpolation: str | int | INTER = INTER.BILINEAR, pad_value: int | tuple[int, int, int] | None = 0, pad_mode: str | int | BORDER = BORDER.CONSTANT, return_scale: bool = False) -> np.ndarray | tuple[np.ndarray, float]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **Description**: Resizes an image to fit within `(max_h, max_w)`, and pads to the fixed target size when needed.

- **Parameters**

  - **img** (`np.ndarray`): Input image.
  - **max_h** (`int`): Max output height (and the fixed padded height).
  - **max_w** (`int`): Max output width (and the fixed padded width).
  - **interpolation** (`str | int | INTER`): Resize interpolation. Default is `INTER.BILINEAR`.
  - **pad_value** (`int | tuple[int, int, int] | None`): Padding value. For 3-channel images, you can pass an int or a tuple (OpenCV convention: BGR). Default is 0.
  - **pad_mode** (`str | int | BORDER`): Padding mode. Default is `BORDER.CONSTANT`.
  - **return_scale** (`bool`): Whether to return the resize scale. Default is `False`.

- **Returns**

  - When `return_scale=False`: returns `np.ndarray`.
  - When `return_scale=True`: returns `(np.ndarray, float)`, where float is `scale = min(max_h/raw_h, max_w/raw_w)`.

- **Notes**

  - Padding is applied to bottom and right only (top=0, left=0).
  - The image will be downscaled when `max_h/max_w` is smaller than the original, and upscaled when larger.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')

  out, scale = cb.imresize_and_pad_if_need(
      img,
      max_h=640,
      max_w=640,
      pad_value=0,
      return_scale=True,
  )
  print(out.shape, scale)
  ```
