# imbinarize

> [imbinarize(img: np.ndarray, threth: int = cv2.THRESH_BINARY, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L336)

- **Description**: Performs binarization on the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to binarize. If the input image has 3 channels, the function will automatically apply the `bgr2gray` function.
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
