# img_to_md5

> [img_to_md5(img: np.ndarray) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L42)

- **Description**: Generates an MD5 hash based on the given image. The reason for this function's existence is the same as `gen_md5`, but it specifically handles image inputs.

- **Parameters**

  - **img** (`np.ndarray`): The image.

- **Returns**

  - **str**: The MD5 hash.

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  md5 = cb.img_to_md5(img)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
