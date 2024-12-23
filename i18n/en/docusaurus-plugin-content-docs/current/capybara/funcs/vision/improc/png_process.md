# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L80)

- **Description**: Encodes a NumPy image array into a PNG format byte string.

- **Parameters**:

  - **img** (`np.ndarray`): The image array to encode.
  - **compression** (`int`): The compression level, ranging from 0 to 9. 0 means no compression, and 9 represents maximum compression. Default is 1.

- **Return value**:

  - **bytes**: The PNG format byte string after encoding.

- **Example**:

  ```python
  import numpy as np
  import capybara as cb

  img_array = np.random.rand(100, 100, 3) * 255
  encoded_bytes = cb.pngencode(img_array, compression=9)
  ```

## pngdecode

> [pngdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L91)

- **Description**: Decodes a PNG format byte string into a NumPy image array.

- **Parameters**:

  - **byte\_** (`bytes`): The PNG format byte string to decode.

- **Return value**:

  - **np.ndarray**: The image array after decoding.

- **Example**:

  ```python
  decoded_img = cb.pngdecode(encoded_bytes)
  ```
