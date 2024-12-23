# imencode

> [imencode(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L100)

- **Description**: Encodes a NumPy image array into a byte string in a specified format.

- **Parameters**:

  - **img** (`np.ndarray`): The image array to encode.
  - **IMGTYP** (`Union[str, int, IMGTYP]`): The image type. Supported types are `IMGTYP.JPEG` and `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.

- **Return value**:

  - **bytes**: The encoded image byte string.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  encoded_bytes = cb.imencode(img, IMGTYP=cb.IMGTYP.PNG)
  ```
