---
sidebar_position: 8
---

# img_to_md5

> [img_to_md5(img: np.ndarray) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L42)

- **Description**: Generates an MD5 hash based on the given image. This function serves the same purpose as `gen_md5`, but is specifically designed for image inputs.

- **Parameters**:
    - **img** (`np.ndarray`): The image.

- **Returns**:
    - **str**: The MD5 hash.

- **Example**:

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    md5 = D.img_to_md5(img)
    print(md5)
    # >>> 'd41d8cd98f00b204e9800998ecf8427e'
    ```
