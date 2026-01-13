# imdecode

> [imdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Decodes image bytes into a numpy image.

- **Parameters**

  - **byte_** (`bytes`): Image bytes.

- **Returns**

  - **np.ndarray | None**: Decoded image; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import IMGTYP, imdecode, imencode, imread

  img = imread('lena.png')
  encoded_bytes = imencode(img, imgtyp=IMGTYP.PNG)
  decoded_img = imdecode(encoded_bytes)
  ```
