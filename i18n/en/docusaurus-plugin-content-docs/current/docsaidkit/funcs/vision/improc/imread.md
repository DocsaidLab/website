---
sidebar_position: 1
---

# imread

>[imread(path: Union[str, Path], color_base: str = 'BGR', verbose: bool = False) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L197C1-L242C15)

- **Description**: Read an image based on different image formats using different methods. Supported formats and methods are as follows:
  - `.heic`: Read using `read_heic_to_numpy` and convert to `BGR` format.
  - `.jpg`: Read using `jpgread` and convert to `BGR` format.
  - Other formats: Read using `cv2.imread` and convert to `BGR` format.
  - If `jpgread` returns `None`, fall back to using `cv2.imread`.

- **Parameters**
    - **path** (`Union[str, Path]`): The path of the image to be read.
    - **color_base** (`str`): The color space of the image. If not `BGR`, conversion will be done using the `imcvtcolor` function. Default is `BGR`.
    - **verbose** (`bool`): If set to True, a warning will be issued when the read image is None. Default is False.

- **Returns**
    - **np.ndarray**: Returns the numpy ndarray of the image if successful, otherwise returns None.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    ```
