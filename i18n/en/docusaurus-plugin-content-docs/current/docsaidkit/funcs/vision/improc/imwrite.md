---
sidebar_position: 2
---

# imwrite

> [imwrite(img: np.ndarray, path: Union[str, Path] = None, color_base: str = 'BGR', suffix: str = '.jpg') -> bool](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L245C1-L272C67)

- **Description**: Write an image to a file with an option to convert color space. If no path is given, write to a temporary file.

- **Parameters**
    - **img** (`np.ndarray`): The image to be written, represented as a numpy ndarray.
    - **path** (`Union[str, Path]`): The path to write the image file. If None, write to a temporary file. Default is None.
    - **color_base** (`str`): The current color space of the image. If not `BGR`, the function will attempt to convert it to `BGR`. Default is `BGR`.
    - **suffix** (`str`): The suffix for the temporary file if path is None. Default is `.jpg`.

- **Returns**
    - **bool**: Returns True if the write operation is successful, otherwise returns False.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    D.imwrite(img, 'lena.jpg')
    ```
