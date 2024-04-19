---
sidebar_position: 8
---

# imdecode

> [imdecode(byte_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L107)

- **Description**: Decode an image byte string to a NumPy image array.

- **Parameters**
    - **byte_** (`bytes`): The image byte string to decode.

- **Returns**
    - **np.ndarray**: The decoded image array.

- **Example**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    encoded_bytes = D.imencode(img, IMGTYP=D.IMGTYP.PNG)
    decoded_img = D.imdecode(encoded_bytes)
    ```
