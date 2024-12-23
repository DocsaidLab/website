# imresize

> [imresize(img: np.ndarray, size: Tuple[int, int], interpolation: Union[str, int, INTER] = INTER.BILINEAR, return_scale: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L15)

- **Description**: Resizes the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be resized.
  - **size** (`Tuple[int, int]`): The target size of the resized image. If only one dimension is given, the other dimension is calculated to maintain the original aspect ratio.
  - **interpolation** (`Union[str, int, INTER]`): The interpolation method. Available options include: INTER.NEAREST, INTER.LINEAR, INTER.CUBIC, INTER.LANCZOS4. Default is INTER.LINEAR.
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
