# imwrite

> [imwrite(img: np.ndarray, path: str | Path | None = None, color_base: str = 'BGR', suffix: str = '.jpg') -> bool](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Writes an image to a file, with optional color base conversion. If `path` is `None`, it writes to a temporary file.

- **Parameters**

  - **img** (`np.ndarray`): Image array.
  - **path** (`str | Path | None`): Output path. If `None`, writes to a temporary file. Default is `None`.
  - **color_base** (`str`): Current color base of `img`. If not `BGR`, it converts to `BGR` before writing. Default is `BGR`.
  - **suffix** (`str`): Temp file suffix when `path` is `None`. Default is `.jpg`.

- **Returns**

  - **bool**: Whether the write succeeded.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  cb.imwrite(img, 'lena.jpg')
  ```
