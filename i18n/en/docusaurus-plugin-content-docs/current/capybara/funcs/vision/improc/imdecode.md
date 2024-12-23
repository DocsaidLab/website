# imdecode

> [imdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L107)

- **Description**: Decodes an image byte string into a NumPy image array.

- **Parameters**:

  - **byte\_** (`bytes`): The image byte string to decode.

- **Return value**:

  - **np.ndarray**: The decoded image array.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  encoded_bytes = cb.imencode(img, IMGTYP=cb.IMGTYP.PNG)
  decoded_img = cb.imdecode(encoded_bytes)
  ```
