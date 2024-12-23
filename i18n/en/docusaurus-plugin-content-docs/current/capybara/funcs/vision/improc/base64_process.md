# Base64 Process

`pybase64` is a Python library that provides Base64 encoding and decoding functionalities. It supports various encoding formats, including standard Base64, Base64 URL, and Base64 URL file name safe encoding. `pybase64` is an enhanced version of the `base64` module, offering more features and options.

In image processing, we often need to convert image data into Base64 encoded strings for use in web transmission. `pybase64` provides a convenient interface for fast Base64 encoding and decoding operations, supporting various encoding formats to meet different needs.

- **Common Issue: String vs Byte String?**

  In Python, a string is a sequence of Unicode characters, while a byte string is a sequence of "bytes." In Base64 encoding, we typically use byte strings for encoding and decoding, as Base64 encoding operates on byte data.

## img_to_b64

> [img_to_b64(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L116)

- **Description**: Converts a NumPy image array into a Base64 byte string.

- **Parameters**:

  - **img** (`np.ndarray`): The image array to convert.
  - **IMGTYP** (`Union[str, int, IMGTYP]`): The image type. Supported types are `IMGTYP.JPEG` and `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.

- **Return value**:

  - **bytes**: The Base64 encoded byte string.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  b64 = cb.img_to_b64(img, IMGTYP=cb.IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L126)

- **Description**: Converts a NumPy array into a Base64 byte string.

- **Parameters**:

  - **x** (`np.ndarray`): The NumPy array to convert.
  - **dtype** (`str`): The data type. Default is `'float32'`.

- **Return value**:

  - **bytes**: The Base64 encoded byte string.

- **Example**:

  ```python
  import capybara as cb
  import numpy as np

  x = np.random.rand(100, 100, 3)
  b64 = cb.npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L130)

- **Description**: Converts a NumPy array into a Base64 string.

- **Parameters**:

  - **x** (`np.ndarray`): The NumPy array to convert.
  - **dtype** (`str`): The data type. Default is `'float32'`.
  - **string_encode** (`str`): The string encoding. Default is `'utf-8'`.

- **Return value**:

  - **str**: The Base64 encoded string.

- **Example**:

  ```python
  import capybara as cb
  import numpy as np

  x = np.random.rand(100, 100, 3)

  b64str = cb.npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG, string_encode: str = 'utf-8') -> Union[str, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L134)

- **Description**: Converts a NumPy image array into a Base64 string.

- **Parameters**:

  - **img** (`np.ndarray`): The image array to convert.
  - **IMGTYP** (`Union[str, int, IMGTYP]`): The image type. Supported types are `IMGTYP.JPEG` and `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.
  - **string_encode** (`str`): The string encoding. Default is `'utf-8'`.

- **Return value**:

  - **str**: The Base64 encoded string.

- **Example**:

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  b64str = cb.img_to_b64str(img, IMGTYP=cb.IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L143)

- **Description**: Converts a Base64 byte string into a NumPy image array.

- **Parameters**:

  - **b64** (`bytes`): The Base64 byte string to convert.

- **Return value**:

  - **np.ndarray**: The converted NumPy image array.

- **Example**:

  ```python
  import capybara as cb

  b64 = cb.img_to_b64(cb.imread('lena.png'))
  img = cb.b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: Union[str, None], string_encode: str = 'utf-8') -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L151)

- **Description**: Converts a Base64 string into a NumPy image array.

- **Parameters**:

  - **b64str** (`Union[str, None]`): The Base64 string to convert.
  - **string_encode** (`str`): The string encoding. Default is `'utf-8'`.

- **Return value**:

  - **np.ndarray**: The converted NumPy image array.

- **Example**:

  ```python
  import capybara as cb

  b64 = cb.img_to_b64(cb.imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = cb.b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L166)

- **Description**: Converts a Base64 byte string into a NumPy array.

- **Parameters**:

  - **x** (`bytes`): The Base64 byte string to convert.
  - **dtype** (`str`): The data type. Default is `'float32'`.

- **Return value**:

  - **np.ndarray**: The converted NumPy array.

- **Example**:

  ```python
  import capybara as cb

  b64 = cb.npy_to_b64(np.random.rand(100, 100, 3))
  x = cb.b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: bytes, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L170)

- **Description**: Converts a Base64 string into a NumPy array.

- **Parameters**:

  - **x** (`bytes`): The Base64 string to convert.
  - **dtype** (`str`): The data type. Default is `'float32'`.
  - **string_encode** (`str`): The string encoding. Default is `'utf-8'`.

- **Return value**:

  - **np.ndarray**: The converted NumPy array.

- **Example**:

  ```python
  import capybara as cb

  b64 = cb.npy_to_b64(np.random.rand(100, 100, 3))
  x = cb.b64str_to_npy(b64.decode('utf-8'))
  ```
