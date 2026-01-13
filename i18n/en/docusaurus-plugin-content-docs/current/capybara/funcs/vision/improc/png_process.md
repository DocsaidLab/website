# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Encodes a numpy image into PNG bytes.

- **Parameters**

  - **img** (`np.ndarray`): Image array.
  - **compression** (`int`): Compression level (0 to 9). Default is 1.

- **Returns**

  - **bytes | None**: PNG bytes; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import imread, pngencode

  img = imread('lena.png')
  encoded_bytes = pngencode(img, compression=9)
  ```

## pngdecode

> [pngdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Decodes PNG bytes into a numpy image.

- **Parameters**

  - **byte_** (`bytes`): PNG bytes.

- **Returns**

  - **np.ndarray | None**: Decoded image; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import pngdecode

  decoded_img = pngdecode(encoded_bytes)
  ```
