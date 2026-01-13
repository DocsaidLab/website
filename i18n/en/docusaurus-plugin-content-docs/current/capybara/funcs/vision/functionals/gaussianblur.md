# gaussianblur

> [gaussianblur(img: np.ndarray, ksize: int | tuple[int, int] | np.ndarray = 3, sigma_x: int = 0, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **Description**: Applies a Gaussian blur to the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be blurred.
  - **ksize** (`Union[int, Tuple[int, int]]`): The size of the kernel used for the blur. If an integer is provided, a square kernel of the specified size is used. If a tuple `(k_height, k_width)` is provided, a rectangular kernel of the specified size is used. Default is 3.
  - **sigma_x** (`int`): The standard deviation in the X direction for the Gaussian kernel. Default is 0.

- **Notes**

  - For backward compatibility, `sigmaX=...` is also accepted.

- **Returns**

  - **np.ndarray**: The blurred image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  blur_img = cb.gaussianblur(img, ksize=5)
  ```

  ![gaussianblur](./resource/test_gaussianblur.jpg)
