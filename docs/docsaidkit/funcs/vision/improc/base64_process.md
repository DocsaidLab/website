---
sidebar_position: 6
---

# Base64 Process

`pybase64` 是一個 Python 函式庫，提供了 Base64 編碼和解碼的功能。 它支援多種編碼格式，包括標準 Base64、Base64 URL 和 Base64 URL 檔案名稱安全編碼。 `pybase64` 是基於 `base64` 模組的增強版本，提供了更多的功能和選項。

在影像處理中，我們經常需要將影像資料轉換為 Base64 編碼的字串，以便在網路傳輸中使用。 `pybase64` 提供了方便的介面，可以快速地進行 Base64 編碼和解碼操作，同時支援多種編碼格式，滿足不同的需求。

- **常見問題：字串 vs 位元組字串？**

  在 Python 中，字串（string）是 Unicode 字元序列，而位元組字串（bytes）是「位元組」序列。 在 Base64 編碼中，我們通常使用「位元組」字串進行編碼和解碼操作，因為 Base64 編碼是對「位元組」資料進行的。

## img_to_b64

> [img_to_b64(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L116)

- **說明**：將 NumPy 圖像數組轉換為 Base64 字節串。

- **參數**

  - **img** (`np.ndarray`)：要轉換的圖像數組。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：圖像類型。支持的類型有 `IMGTYP.JPEG` 和 `IMGTYP.PNG`。預設為 `IMGTYP.JPEG`。

- **傳回值**

  - **bytes**：轉換後的 Base64 字節串。

- **範例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  b64 = D.img_to_b64(img, IMGTYP=D.IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L126)

- **說明**：將 NumPy 數組轉換為 Base64 字節串。

- **參數**

  - **x** (`np.ndarray`)：要轉換的 NumPy 數組。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。

- **傳回值**

  - **bytes**：轉換後的 Base64 字節串。

- **範例**

  ```python
  import docsaidkit as D
  import numpy as np

  x = np.random.rand(100, 100, 3)
  b64 = D.npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L130)

- **說明**：將 NumPy 數組轉換為 Base64 字串。

- **參數**

  - **x** (`np.ndarray`)：要轉換的 NumPy 數組。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **str**：轉換後的 Base64 字串。

- **範例**

  ```python
  import docsaidkit as D
  import numpy as np

  x = np.random.rand(100, 100, 3)

  b64str = D.npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG, string_encode: str = 'utf-8') -> Union[str, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L134)

- **說明**：將 NumPy 圖像數組轉換為 Base64 字串。

- **參數**

  - **img** (`np.ndarray`)：要轉換的圖像數組。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：圖像類型。支持的類型有 `IMGTYP.JPEG` 和 `IMGTYP.PNG`。預設為 `IMGTYP.JPEG`。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **str**：轉換後的 Base64 字串。

- **範例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  b64str = D.img_to_b64str(img, IMGTYP=D.IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L143)

- **說明**：將 Base64 字節串轉換為 NumPy 圖像數組。

- **參數**

  - **b64** (`bytes`)：要轉換的 Base64 字節串。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 圖像數組。

- **範例**

  ```python
  import docsaidkit as D

  b64 = D.img_to_b64(D.imread('lena.png'))
  img = D.b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: Union[str, None], string_encode: str = 'utf-8') -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L151)

- **說明**：將 Base64 字串轉換為 NumPy 圖像數組。

- **參數**

  - **b64str** (`Union[str, None]`)：要轉換的 Base64 字串。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 圖像數組。

- **範例**

  ```python
  import docsaidkit as D

  b64 = D.img_to_b64(D.imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = D.b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L166)

- **說明**：將 Base64 字節串轉換為 NumPy 數組。

- **參數**

  - **x** (`bytes`)：要轉換的 Base64 字節串。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 數組。

- **範例**

  ```python
  import docsaidkit as D

  b64 = D.npy_to_b64(np.random.rand(100, 100, 3))
  x = D.b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: bytes, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L170)

- **說明**：將 Base64 字串轉換為 NumPy 數組。

- **參數**

  - **x** (`bytes`)：要轉換的 Base64 字串。
  - **dtype** (`str`)：數據類型。預設為 `'float32'`。
  - **string_encode** (`str`)：字串編碼。預設為 `'utf-8'`。

- **傳回值**

  - **np.ndarray**：轉換後的 NumPy 數組。

- **範例**

  ```python
  import docsaidkit as D

  b64 = D.npy_to_b64(np.random.rand(100, 100, 3))
  x = D.b64str_to_npy(b64.decode('utf-8'))
  ```
