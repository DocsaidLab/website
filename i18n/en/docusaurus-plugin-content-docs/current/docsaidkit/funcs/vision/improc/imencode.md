---
sidebar_position: 7
---

# imencode

> [imencode(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L100)

- **Description**: Encode a NumPy image array into a byte string in the specified format.

- **Parameters**
    - **img** (`np.ndarray`): The image array to encode.
    - **IMGTYP** (`Union[str, int, IMGTYP]`): The type of the image. Supported types are `IMGTYP.JPEG` and `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.

- **Returns**
    - **bytes**: The encoded image byte string.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    encoded_bytes = D.imencode(img, IMGTYP=D.IMGTYP.PNG)
    ```
