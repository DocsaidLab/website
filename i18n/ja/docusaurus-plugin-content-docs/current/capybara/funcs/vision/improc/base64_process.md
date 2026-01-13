# Base64 Process

このモジュールは内部で `pybase64` を使って Base64 のエンコード／デコードを行い、`bytes` と `str` の両方のインターフェース（例：`img_to_b64` と `img_to_b64str`）を提供します。

- **よくある疑問：文字列（str） vs バイト列（bytes）？**

  Python の文字列（str）は Unicode 文字列で、バイト列（bytes）は「バイト」の列です。Base64 はバイト列を対象にエンコード／デコードするため、基本的には bytes ベースで扱います。

## img_to_b64

> [img_to_b64(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, **kwargs: object) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy の画像配列を Base64 の bytes に変換します。

- **パラメータ**

  - **img** (`np.ndarray`)：変換する画像配列。
  - **imgtyp** (`str | int | IMGTYP`)：画像タイプ。`IMGTYP.JPEG` / `IMGTYP.PNG` を指定できます。デフォルトは `IMGTYP.JPEG`。

- **戻り値**

  - **bytes | None**：Base64 bytes。エンコード失敗時は `None`。

- **備考**

  - 互換性のため `IMGTYP=...` も受け付けます（`imgtyp` と同時に渡すと `TypeError`）。

- **例**

  ```python
  from capybara.vision.improc import IMGTYP, img_to_b64, imread

  img = imread('lena.png')
  b64 = img_to_b64(img, imgtyp=IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy 配列を Base64 bytes に変換します。

- **パラメータ**

  - **x** (`np.ndarray`)：変換する NumPy 配列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。

- **戻り値**

  - **bytes**：Base64 bytes。

- **例**

  ```python
  import numpy as np
  from capybara.vision.improc import npy_to_b64

  x = np.random.rand(100, 100, 3)
  b64 = npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy 配列を Base64 文字列に変換します。

- **パラメータ**

  - **x** (`np.ndarray`)：変換する NumPy 配列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **戻り値**

  - **str**：Base64 文字列。

- **例**

  ```python
  import numpy as np
  from capybara.vision.improc import npy_to_b64str

  x = np.random.rand(100, 100, 3)
  b64str = npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, string_encode: str = 'utf-8', **kwargs: object) -> str | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：NumPy の画像配列を Base64 文字列に変換します。

- **パラメータ**

  - **img** (`np.ndarray`)：変換する画像配列。
  - **imgtyp** (`str | int | IMGTYP`)：画像タイプ。`IMGTYP.JPEG` / `IMGTYP.PNG` を指定できます。デフォルトは `IMGTYP.JPEG`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **戻り値**

  - **str | None**：Base64 文字列。エンコード失敗時は `None`。

- **備考**

  - 互換性のため `IMGTYP=...` も受け付けます（`imgtyp` と同時に渡すと `TypeError`）。

- **例**

  ```python
  from capybara import IMGTYP, img_to_b64str, imread

  img = imread('lena.png')
  b64str = img_to_b64str(img, imgtyp=IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：Base64 bytes を NumPy 画像配列に変換します。

- **パラメータ**

  - **b64** (`bytes`)：Base64 bytes。

- **戻り値**

  - **np.ndarray | None**：NumPy 画像配列。失敗時は `None`。

- **例**

  ```python
  from capybara.vision.improc import b64_to_img, img_to_b64, imread

  b64 = img_to_b64(imread('lena.png'))
  img = b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: str | None, string_encode: str = 'utf-8') -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：Base64 文字列を NumPy 画像配列に変換します。

- **パラメータ**

  - **b64str** (`str | None`)：Base64 文字列。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **戻り値**

  - **np.ndarray | None**：NumPy 画像配列。失敗時は `None`。

- **例**

  ```python
  from capybara.vision.improc import b64str_to_img, img_to_b64, imread

  b64 = img_to_b64(imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：Base64 bytes を NumPy 配列に変換します。

- **パラメータ**

  - **x** (`bytes`)：Base64 bytes。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。

- **戻り値**

  - **np.ndarray**：NumPy 配列。

- **例**

  ```python
  import numpy as np
  from capybara.vision.improc import b64_to_npy, npy_to_b64

  b64 = npy_to_b64(np.random.rand(100, 100, 3))
  x = b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: str, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **説明**：Base64 文字列を NumPy 配列に変換します。

- **パラメータ**

  - **x** (`str`)：Base64 文字列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **戻り値**

  - **np.ndarray**：NumPy 配列。

- **例**

  ```python
  import numpy as np
  from capybara.vision.improc import b64str_to_npy, npy_to_b64

  b64 = npy_to_b64(np.random.rand(100, 100, 3))
  x = b64str_to_npy(b64.decode('utf-8'))
  ```

