# Base64 Process

This module uses `pybase64` internally and provides both `bytes` and `str` interfaces (e.g. `img_to_b64` vs `img_to_b64str`).

- **Common issue: string vs bytes?**

  In Python, `str` is a sequence of Unicode characters, while `bytes` is a sequence of raw bytes. Base64 operates on bytes, so encoding/decoding typically happens on `bytes`. String variants are provided for convenience.

## img_to_b64

> [img_to_b64(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, **kwargs: object) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Encodes a numpy image into Base64 `bytes`.

- **Parameters**

  - **img** (`np.ndarray`): Image array.
  - **imgtyp** (`str | int | IMGTYP`): Image type. Supports `IMGTYP.JPEG` / `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.

- **Returns**

  - **bytes | None**: Base64 bytes; returns `None` when encoding fails.

- **Notes**

  - For backward compatibility, `IMGTYP=...` is also accepted (providing both `imgtyp` and `IMGTYP` raises `TypeError`).

- **Example**

  ```python
  from capybara.vision.improc import IMGTYP, img_to_b64, imread

  img = imread('lena.png')
  b64 = img_to_b64(img, imgtyp=IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Encodes a numpy array into Base64 `bytes`.

- **Parameters**

  - **x** (`np.ndarray`): Numpy array.
  - **dtype** (`str`): Cast dtype before encoding. Default is `'float32'`.

- **Returns**

  - **bytes**: Base64 bytes.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.improc import npy_to_b64

  x = np.random.rand(100, 100, 3)
  b64 = npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Encodes a numpy array into a Base64 string.

- **Parameters**

  - **x** (`np.ndarray`): Numpy array.
  - **dtype** (`str`): Cast dtype before encoding. Default is `'float32'`.
  - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**

  - **str**: Base64 string.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.improc import npy_to_b64str

  x = np.random.rand(100, 100, 3)
  b64str = npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, string_encode: str = 'utf-8', **kwargs: object) -> str | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Encodes a numpy image into a Base64 string.

- **Parameters**

  - **img** (`np.ndarray`): Image array.
  - **imgtyp** (`str | int | IMGTYP`): Image type. Supports `IMGTYP.JPEG` / `IMGTYP.PNG`. Default is `IMGTYP.JPEG`.
  - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**

  - **str | None**: Base64 string; returns `None` when encoding fails.

- **Notes**

  - For backward compatibility, `IMGTYP=...` is also accepted (providing both `imgtyp` and `IMGTYP` raises `TypeError`).

- **Example**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  b64str = cb.img_to_b64str(img, imgtyp=cb.IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Decodes Base64 bytes into a numpy image.

- **Parameters**

  - **b64** (`bytes`): Base64 bytes.

- **Returns**

  - **np.ndarray | None**: Decoded image; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import b64_to_img, img_to_b64, imread

  b64 = img_to_b64(imread('lena.png'))
  img = b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: str | None, string_encode: str = 'utf-8') -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Decodes a Base64 string into a numpy image.

- **Parameters**

  - **b64str** (`str | None`): Base64 string.
  - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**

  - **np.ndarray | None**: Decoded image; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import b64str_to_img, img_to_b64, imread

  b64 = img_to_b64(imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Decodes Base64 bytes into a numpy array.

- **Parameters**

  - **x** (`bytes`): Base64 bytes.
  - **dtype** (`str`): Output dtype. Default is `'float32'`.

- **Returns**

  - **np.ndarray**: Decoded numpy array.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.improc import b64_to_npy, npy_to_b64

  b64 = npy_to_b64(np.random.rand(100, 100, 3))
  x = b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: str, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Decodes a Base64 string into a numpy array.

- **Parameters**

  - **x** (`str`): Base64 string.
  - **dtype** (`str`): Output dtype. Default is `'float32'`.
  - **string_encode** (`str`): String encoding. Default is `'utf-8'`.

- **Returns**

  - **np.ndarray**: Decoded numpy array.

- **Example**

  ```python
  import numpy as np
  from capybara.vision.improc import b64str_to_npy, npy_to_b64

  b64 = npy_to_b64(np.random.rand(100, 100, 3))
  x = b64str_to_npy(b64.decode('utf-8'))
  ```
