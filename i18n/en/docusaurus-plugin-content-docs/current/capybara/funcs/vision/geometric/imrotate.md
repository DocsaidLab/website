# imrotate

> [imrotate(img: np.ndarray, angle: float, scale: float = 1, interpolation: str | int | INTER = INTER.BILINEAR, bordertype: str | int | BORDER = BORDER.CONSTANT, bordervalue: int | tuple[int, ...] | None = None, expand: bool = True, center: tuple[int, int] | None = None) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/geometric.py)

- **Description**: Rotates the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be rotated.
  - **angle** (`float`): The rotation angle in degrees, counterclockwise.
  - **scale** (`float`): The scaling factor. Default is 1.
  - **interpolation** (`str | int | INTER`): Interpolation method. Available options: `INTER.NEAREST`, `INTER.BILINEAR`, `INTER.CUBIC`, `INTER.AREA`, `INTER.LANCZOS4`. Default is `INTER.BILINEAR`.
  - **bordertype** (`Union[str, int, BORDER]`): The border type. Available options include: BORDER.CONSTANT, BORDER.REPLICATE, BORDER.REFLECT, BORDER.REFLECT_101. Default is BORDER.CONSTANT.
  - **bordervalue** (`Union[int, Tuple[int, int, int]]`): The value used to fill the border. Only used when bordertype is BORDER.CONSTANT. Default is None.
  - **expand** (`bool`): Whether to expand the output image to fit the entire rotated image. Default is `True`.
  - **center** (`tuple[int, int] | None`): The center of rotation. `None` means image center.

- **Returns**

  - **np.ndarray**: The rotated image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  rotate_img = cb.imrotate(img, 45, bordertype=cb.BORDER.CONSTANT, expand=True)

  # Resize the rotated image to the original size for visualization
  rotate_img = cb.imresize(rotate_img, [img.shape[0], img.shape[1]])
  ```

  ![imrotate](./resource/test_imrotate.jpg)
