# imwrite

> [imwrite(img: np.ndarray, path: Union[str, Path] = None, color_base: str = 'BGR', suffix: str = '.jpg') -> bool](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L245)

- **Description**: Writes an image to a file, with an optional color space conversion. If no path is provided, the image is written to a temporary file.

- **Parameters**:

  - **img** (`np.ndarray`): The image to write, represented as a NumPy ndarray.
  - **path** (`Union[str, Path]`): The path where the image file will be saved. If `None`, a temporary file will be created. Default is `None`.
  - **color_base** (`str`): The current color space of the image. If it is not `BGR`, the function will attempt to convert it to `BGR`. Default is `BGR`.
  - **suffix** (`str`): The suffix for the temporary file if `path` is `None`. Default is `.jpg`.

- **Return value**:

  - **bool**: Returns `True` if the write operation is successful, otherwise returns `False`.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  cb.imwrite(img, 'lena.jpg')
  ```
