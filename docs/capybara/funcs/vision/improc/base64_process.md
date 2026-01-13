# Base64 Process

此模組內部使用 `pybase64` 進行 Base64 編碼／解碼，並提供 `bytes` 與 `str` 兩種介面（例如 `img_to_b64` 與 `img_to_b64str`）。

- **常見問題：字串 vs 位元組字串？**

  在 Python 中，字串（string）是 Unicode 字元序列，而位元組字串（bytes）是「位元組」序列。 在 Base64 編碼中，我們通常使用「位元組」字串進行編碼和解碼操作，因為 Base64 編碼是對「位元組」資料進行的。

## img_to_b64

> [img_to_b64(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, **kwargs: object) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 NumPy 圖像數組轉換為 Base64 字節串。

- **參數**

  - **img** (`np.ndarray`)：要轉換的圖像數組。
  - **imgtyp** (`str | int | IMGTYP`)：圖像類型。支援 `IMGTYP.JPEG` / `IMGTYP.PNG`。預設為 `IMGTYP.JPEG`。

- **傳回值**

  - **bytes | None**：轉換後的 Base64 字節串；編碼失敗時回傳 `None`。

- **備註**

  - 為了兼容舊版呼叫方式，亦接受 `IMGTYP=...`（與 `imgtyp` 同時提供會拋出 `TypeError`）。

- **範例**

  ```python
  from capybara.vision.improc import IMGTYP, img_to_b64, imread

  img = imread('lena.png')
  b64 = img_to_b64(img, imgtyp=IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 NumPy 數組轉換為 Base64 字節串。

- **參數**

  - **x** (`np.ndarray`)：要轉換的 NumPy 數組。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。

- **傳回值**

  - **bytes**：轉換後的 Base64 字節串。

- **範例**

  ```python
  import numpy as np
  from capybara.vision.improc import npy_to_b64

  x = np.random.rand(100, 100, 3)
  b64 = npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 NumPy 數組轉換為 Base64 字串。

- **參數**

  - **x** (`np.ndarray`)：要轉換的 NumPy 數組。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **str**：轉換後的 Base64 字串。

- **範例**

  ```python
  import numpy as np
  from capybara.vision.improc import npy_to_b64str

  x = np.random.rand(100, 100, 3)

  b64str = npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, imgtyp: str | int | IMGTYP = IMGTYP.JPEG, string_encode: str = 'utf-8', **kwargs: object) -> str | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 NumPy 圖像數組轉換為 Base64 字串。

- **參數**

  - **img** (`np.ndarray`)：要轉換的圖像數組。
  - **imgtyp** (`str | int | IMGTYP`)：圖像類型。支援 `IMGTYP.JPEG` / `IMGTYP.PNG`。預設為 `IMGTYP.JPEG`。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **str | None**：轉換後的 Base64 字串；編碼失敗時回傳 `None`。

- **備註**

  - 為了兼容舊版呼叫方式，亦接受 `IMGTYP=...`（與 `imgtyp` 同時提供會拋出 `TypeError`）。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  b64str = cb.img_to_b64str(img, imgtyp=cb.IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 Base64 字節串轉換為 NumPy 圖像數組。

- **參數**

  - **b64** (`bytes`)：要轉換的 Base64 字節串。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 圖像數組。

- **範例**

  ```python
  from capybara.vision.improc import b64_to_img, img_to_b64, imread

  b64 = img_to_b64(imread('lena.png'))
  img = b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: str | None, string_encode: str = 'utf-8') -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 Base64 字串轉換為 NumPy 圖像數組。

- **參數**

  - **b64str** (`Union[str, None]`)：要轉換的 Base64 字串。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 圖像數組。

- **範例**

  ```python
  from capybara.vision.improc import b64str_to_img, img_to_b64, imread

  b64 = img_to_b64(imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 Base64 字節串轉換為 NumPy 數組。

- **參數**

  - **x** (`bytes`)：要轉換的 Base64 字節串。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 數組。

- **範例**

  ```python
  import numpy as np
  from capybara.vision.improc import b64_to_npy, npy_to_b64

  b64 = npy_to_b64(np.random.rand(100, 100, 3))
  x = b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: str, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **說明**：將 Base64 字串轉換為 NumPy 數組。

- **參數**

  - **x** (`str`)：要轉換的 Base64 字串。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 數組。

- **範例**

  ```python
  import numpy as np
  from capybara.vision.improc import b64str_to_npy, npy_to_b64

  b64 = npy_to_b64(np.random.rand(100, 100, 3))
  x = b64str_to_npy(b64.decode('utf-8'))
  ```
