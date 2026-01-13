# img_to_md5

> [img_to_md5(img: np.ndarray) -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Generates an MD5 hash based on the given image. The reason for this function's existence is the same as `gen_md5`, but it specifically handles image inputs.

- **Parameters**

  - **img** (`np.ndarray`): The image.

- **Returns**

  - **str**: The MD5 hash.

- **Example**

  ```python
  from capybara import imread
  from capybara.utils.files_utils import img_to_md5

  img = imread('lena.png')
  if img is None:
      raise RuntimeError('Failed to read image.')
  md5 = img_to_md5(img)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
