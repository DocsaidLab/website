# imread

> [imread(path: Union[str, Path], color_base: str = 'BGR', verbose: bool = False) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L197)

- **Description**: Reads an image using different methods based on the image format. Supported formats are as follows:

  - `.heic`: Uses `read_heic_to_numpy` to read and converts to `BGR` format.
  - `.jpg`: Uses `jpgread` to read and converts to `BGR` format.
  - Other formats: Uses `cv2.imread` to read and converts to `BGR` format.
  - If `jpgread` returns `None`, `cv2.imread` will be used as a fallback.

- **Parameters**:

  - **path** (`Union[str, Path]`): The path to the image file.
  - **color_base** (`str`): The color space of the image. If it is not `BGR`, the function will convert it using the `imcvtcolor` function. Default is `BGR`.
  - **verbose** (`bool`): If set to `True`, a warning will be issued when the image read result is `None`. Default is `False`.

- **Return value**:

  - **np.ndarray**: Returns the image as a NumPy ndarray if successful, otherwise returns `None`.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ```
