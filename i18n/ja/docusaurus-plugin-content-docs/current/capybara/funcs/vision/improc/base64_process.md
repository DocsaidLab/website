# Base64 Process

`pybase64` is a Python library that provides Base64 encoding and decoding functionality. It supports various encoding formats, including standard Base64, Base64 URL, and Base64 URL-safe filenames. `pybase64` is an enhanced version of the `base64` module, offering more features and options.

In image processing, we often need to convert image data into Base64 encoded strings for use in web transmission. `pybase64` provides a convenient interface to quickly perform Base64 encoding and decoding operations, supporting multiple encoding formats to meet various needs.

- **Common Issue: String vs. Byte String?**

  In Python, a string is a sequence of Unicode characters, while a byte string is a sequence of "bytes". In Base64 encoding, we typically work with "byte" strings because Base64 encoding operates on byte data.

## img_to_b64

> [img_to_b64(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L116)

- **説明**：NumPy 画像配列を Base64 バイト列に変換します。

- **パラメータ**

  - **img** (`np.ndarray`)：変換する画像配列。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：画像タイプ。サポートされているタイプは`IMGTYP.JPEG`と`IMGTYP.PNG`です。デフォルトは`IMGTYP.JPEG`。

- **戻り値**

  - **bytes**：変換後の Base64 バイト列。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  b64 = cb.img_to_b64(img, IMGTYP=cb.IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L126)

- **説明**：NumPy 配列を Base64 バイト列に変換します。

- **パラメータ**

  - **x** (`np.ndarray`)：変換する NumPy 配列。
  - **dtype** (`str`)：データ型。デフォルトは`'float32'`。

- **戻り値**

  - **bytes**：変換後の Base64 バイト列。

- **使用例**

  ```python
  import capybara as cb
  import numpy as np

  x = np.random.rand(100, 100, 3)
  b64 = cb.npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L130)

- **説明**：NumPy 配列を Base64 文字列に変換します。

- **パラメータ**

  - **x** (`np.ndarray`)：変換する NumPy 配列。
  - **dtype** (`str`)：データ型。デフォルトは`'float32'`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは`'utf-8'`。

- **戻り値**

  - **str**：変換後の Base64 文字列。

- **使用例**

  ```python
  import capybara as cb
  import numpy as np

  x = np.random.rand(100, 100, 3)

  b64str = cb.npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG, string_encode: str = 'utf-8') -> Union[str, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L134)

- **説明**：NumPy 画像配列を Base64 文字列に変換します。

- **パラメータ**

  - **img** (`np.ndarray`)：変換する画像配列。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：画像タイプ。サポートされているタイプは`IMGTYP.JPEG`と`IMGTYP.PNG`です。デフォルトは`IMGTYP.JPEG`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは`'utf-8'`。

- **戻り値**

  - **str**：変換後の Base64 文字列。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  b64str = cb.img_to_b64str(img, IMGTYP=cb.IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L143)

- **説明**：Base64 バイト列を NumPy 画像配列に変換します。

- **パラメータ**

  - **b64** (`bytes`)：変換する Base64 バイト列。

- **戻り値**

  - **np.ndarray**：変換後の NumPy 画像配列。

- **使用例**

  ```python
  import capybara as cb

  b64 = cb.img_to_b64(cb.imread('lena.png'))
  img = cb.b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: Union[str, None], string_encode: str = 'utf-8') -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L151)

- **説明**：Base64 文字列を NumPy 画像配列に変換します。

- **パラメータ**

  - **b64str** (`Union[str, None]`)：変換する Base64 文字列。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは`'utf-8'`。

- **戻り値**

  - **np.ndarray**：変換後の NumPy 画像配列。

- **使用例**

  ```python
  import capybara as cb

  b64 = cb.img_to_b64(cb.imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = cb.b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L166)

- **説明**：Base64 バイト列を NumPy 配列に変換します。

- **パラメータ**

  - **x** (`bytes`)：変換する Base64 バイト列。
  - **dtype** (`str`)：データ型。デフォルトは`'float32'`。

- **戻り値**

  - **np.ndarray**：変換後の NumPy 配列。

- **使用例**

  ```python
  import capybara as cb

  b64 = cb.npy_to_b64(np.random.rand(100, 100, 3))
  x = cb.b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: bytes, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L170)

- **説明**：Base64 文字列を NumPy 配列に変換します。

- **パラメータ**

  - **x** (`bytes`)：変換する Base64 文字列。
  - **dtype** (`str`)：データ型。デフォルトは`'float32'`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは`'utf-8'`。

- **戻り値**

  - **np.ndarray**：変換後の NumPy 配列。

- **使用例**

  ```python
  import capybara as cb

  b64 = cb.npy_to_b64(np.random.rand(100, 100, 3))
  x = cb.b64str_to_npy(b64.decode('utf-8'))
  ```
