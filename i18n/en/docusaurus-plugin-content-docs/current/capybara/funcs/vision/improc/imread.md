# imread

> [imread(path: str | Path, color_base: str = 'BGR', verbose: bool = False) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Reads an image and returns a BGR numpy image (with optional color conversion).

  - If suffix is `.heic`: reads via `pillow-heif` (outputs BGR).
  - Otherwise: tries `jpgread` first (handles JPEG + EXIF orientation), then falls back to `cv2.imread` on failure.

- **Parameters**

  - **path** (`str | Path`): Image path.
  - **color_base** (`str`): Output color space. If not `BGR`, it converts via `imcvtcolor`. Default is `BGR`.
  - **verbose** (`bool`): If `True`, warns when the decoded image is `None`. Default is `False`.

- **Returns**

  - **np.ndarray | None**: Decoded image, or `None` on failure.

- **Exceptions**

  - **FileExistsError**: When `path` does not exist.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  ```
