# imadjust

> [imadjust(img: np.ndarray, rng_out: Tuple[int, int] = (0, 255), gamma: float = 1.0, color_base: str = 'BGR') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L122)

- **Description**: Adjusts the intensity of an image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to adjust the intensity. It can be 2-D or 3-D.
  - **rng_out** (`Tuple[int, int]`): The target intensity range for the output image. Default is (0, 255).
  - **gamma** (`float`): The value used for gamma correction. If gamma is less than 1, the mapping will be skewed toward higher (brighter) output values. If gamma is greater than 1, the mapping will be skewed toward lower (darker) output values. Default is 1.0 (linear mapping).
  - **color_base** (`str`): The color basis of the input image. Should be 'BGR' or 'RGB'. Default is 'BGR'.

- **Returns**

  - **np.ndarray**: The adjusted image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  adj_img = cb.imadjust(img, gamma=2)
  ```

  ![imadjust](./resource/test_imadjust.jpg)
