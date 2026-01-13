# imresize

> [imresize(img: np.ndarray, size: tuple[int | None, int | None], interpolation: str | int | INTER = INTER.BILINEAR, return_scale: bool = False)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **Description**: Resizes the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be resized.
  - **size** (`tuple[int | None, int | None]`): Target size `(height, width)`. If one dimension is `None`, the other is inferred to keep the aspect ratio.
  - **interpolation** (`str | int | INTER`): Interpolation method. Available options: `INTER.NEAREST`, `INTER.BILINEAR`, `INTER.CUBIC`, `INTER.AREA`, `INTER.LANCZOS4`. Default is `INTER.BILINEAR`.
  - **return_scale** (`bool`): Whether to return the scaling ratio. Default is False.

- **Returns**

  - **np.ndarray**: The resized image.
  - **Tuple[np.ndarray, float, float]**: The resized image along with the width and height scaling ratios.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')

  # Resize the image to H=256, W=256
  resized_img = cb.imresize(img, [256, 256])

  # Resize the image to H=256, keeping the aspect ratio
  resized_img = cb.imresize(img, [256, None])

  # Resize the image to W=256, keeping the aspect ratio
  resized_img = cb.imresize(img, [None, 256])
  ```
