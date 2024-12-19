---
sidebar_position: 5
---

# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L80)

- **Description**: Encode a NumPy image array into a byte string in PNG format.

- **Parameters**:
  - **img** (`np.ndarray`): The image array to be encoded.
  - **compression** (`int`): Compression level, ranging from 0 to 9. 0 means no compression, 9 means highest compression. Default is 1.

- **Returns**
    - **bytes**: The encoded byte string in PNG format.

- **Example**

    ```python
    import numpy as np
    import docsaidkit as D

    img_array = np.random.rand(100, 100, 3) * 255
    encoded_bytes = D.pngencode(img_array, compression=9)
    ```

## pngdecode

> [pngdecode(byte_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L91)

- **Description**: Decode a byte string in PNG format into a NumPy image array.

- **Parameters**:
  - **byte_** (`bytes`): The byte string in PNG format to be decoded.

- **Returns**
    - **np.ndarray**: The decoded image array.

- **Example**

    ```python
    decoded_img = D.pngdecode(encoded_bytes)
    ```
