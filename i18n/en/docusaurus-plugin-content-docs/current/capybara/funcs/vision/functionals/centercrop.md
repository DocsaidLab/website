# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L374)

- **Description**: Performs a center crop on the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be center-cropped.

- **Returns**

  - **np.ndarray**: The cropped image.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  img = cb.imresize(img, [128, 256])
  crop_img = cb.centercrop(img)
  ```

  The green box represents the area of the center crop.

  ![centercrop](./resource/test_centercrop.jpg)
