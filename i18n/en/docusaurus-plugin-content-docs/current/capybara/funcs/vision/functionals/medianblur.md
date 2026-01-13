# medianblur

> [medianblur(img: np.ndarray, ksize: int = 3, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **Description**: Applies median blur to the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be blurred.
  - **ksize** (`int`): Median blur kernel size; must be a positive odd integer. Default is 3.

- **Returns**

  - **np.ndarray**: The blurred image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  blur_img = cb.medianblur(img, ksize=5)
  ```

  ![medianblur](./resource/test_medianblur.jpg)
