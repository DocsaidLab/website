# imbinarize

> [imbinarize(img: np.ndarray, threth: int = cv2.THRESH_BINARY, color_base: str = "BGR") -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **Description**: Performs binarization on the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to binarize. If it has 3 channels, it is converted to grayscale via `imcvtcolor(..., f\"{color_base}2GRAY\")` first.
  - **threth** (`int`): The threshold type. There are two threshold types:
    1. `cv2.THRESH_BINARY`: `cv2.THRESH_OTSU + cv2.THRESH_BINARY`
    2. `cv2.THRESH_BINARY_INV`: `cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV`
  - **color_base** (`str`): The color space of the input image. Default is `'BGR'`.

- **Returns**

  - **np.ndarray**: The binarized image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  bin_img = cb.imbinarize(img)
  ```

  ![imbinarize](./resource/test_imbinarize.jpg)
