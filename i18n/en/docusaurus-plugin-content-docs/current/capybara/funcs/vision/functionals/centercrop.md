# centercrop

> [centercrop(img: np.ndarray) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **Description**: Performs a center crop on the input image.

- **Parameters**

  - **img** (`np.ndarray`): The input image to be center-cropped.

- **Returns**

  - **np.ndarray**: The cropped image.

- **Example**

  ```python
  from capybara import imread, imresize
  from capybara.vision.functionals import centercrop

  img = imread('lena.png')
  img = imresize(img, [128, 256])
  crop_img = centercrop(img)
  ```

  The green box represents the center-crop region.

  ![centercrop](./resource/test_centercrop.jpg)
