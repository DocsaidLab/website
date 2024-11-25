---
sidebar_position: 6
---

# Base64 Process

`pybase64`は、Base64 エンコードおよびデコード機能を提供する Python ライブラリです。標準の Base64、Base64 URL、および Base64 URL ファイル名安全エンコードなど、さまざまなエンコード形式をサポートしています。`pybase64`は、`base64`モジュールをベースにした強化バージョンで、より多くの機能とオプションを提供します。

画像処理では、画像データを Base64 エンコードされた文字列に変換して、インターネットでの転送などで使用することがよくあります。`pybase64`は、さまざまなエンコード形式をサポートし、簡単なインターフェースで Base64 エンコードおよびデコードを迅速に行うことができます。

- **よくある質問：文字列とバイト文字列？**

  Python では、文字列（string）は Unicode 文字のシーケンスであり、バイト文字列（bytes）は「バイト」のシーケンスです。Base64 エンコードでは、通常、エンコードおよびデコード操作に「バイト」文字列を使用します。Base64 エンコードは「バイト」データに対して行われます。

## img_to_b64

> [img_to_b64(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L116)

- **説明**：NumPy 画像配列を Base64 バイト文字列に変換します。

- 引数

  - **img** (`np.ndarray`)：変換する画像配列。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：画像タイプ。サポートされているタイプは `IMGTYP.JPEG` と `IMGTYP.PNG` です。デフォルトは `IMGTYP.JPEG`。

- **返り値**

  - **bytes**：変換された Base64 バイト文字列。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  b64 = D.img_to_b64(img, IMGTYP=D.IMGTYP.PNG)
  ```

## npy_to_b64

> [npy_to_b64(x: np.ndarray, dtype='float32') -> bytes](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L126)

- **説明**：NumPy 配列を Base64 バイト文字列に変換します。

- 引数

  - **x** (`np.ndarray`)：変換する NumPy 配列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。

- **返り値**

  - **bytes**：変換された Base64 バイト文字列。

- **例**

  ```python
  import docsaidkit as D
  import numpy as np

  x = np.random.rand(100, 100, 3)
  b64 = D.npy_to_b64(x)
  ```

## npy_to_b64str

> [npy_to_b64str(x: np.ndarray, dtype='float32', string_encode: str = 'utf-8') -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L130)

- **説明**：NumPy 配列を Base64 文字列に変換します。

- 引数

  - **x** (`np.ndarray`)：変換する NumPy 配列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **返り値**

  - **str**：変換された Base64 文字列。

- **例**

  ```python
  import docsaidkit as D
  import numpy as np

  x = np.random.rand(100, 100, 3)

  b64str = D.npy_to_b64str(x)
  ```

## img_to_b64str

> [img_to_b64str(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG, string_encode: str = 'utf-8') -> Union[str, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L134)

- **説明**：NumPy 画像配列を Base64 文字列に変換します。

- 引数

  - **img** (`np.ndarray`)：変換する画像配列。
  - **IMGTYP** (`Union[str, int, IMGTYP]`)：画像タイプ。サポートされているタイプは `IMGTYP.JPEG` と `IMGTYP.PNG` です。デフォルトは `IMGTYP.JPEG`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **返り値**

  - **str**：変換された Base64 文字列。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  b64str = D.img_to_b64str(img, IMGTYP=D.IMGTYP.PNG)
  ```

## b64_to_img

> [b64_to_img(b64: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L143)

- **説明**：Base64 バイト文字列を NumPy 画像配列に変換します。

- 引数

  - **b64** (`bytes`)：変換する Base64 バイト文字列。

- **返り値**

  - **np.ndarray**：変換された NumPy 画像配列。

- **例**

  ```python
  import docsaidkit as D

  b64 = D.img_to_b64(D.imread('lena.png'))
  img = D.b64_to_img(b64)
  ```

## b64str_to_img

> [b64str_to_img(b64str: Union[str, None], string_encode: str = 'utf-8') -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L151)

- **説明**：Base64 文字列を NumPy 画像配列に変換します。

- 引数

  - **b64str** (`Union[str, None]`)：変換する Base64 文字列。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **返り値**

  - **np.ndarray**：変換された NumPy 画像配列。

- **例**

  ```python
  import docsaidkit as D

  b64 = D.img_to_b64(D.imread('lena.png'))
  b64str = b64.decode('utf-8')
  img = D.b64str_to_img(b64str)
  ```

## b64_to_npy

> [b64_to_npy(x: bytes, dtype='float32') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L166)

- **説明**：Base64 バイト文字列を NumPy 配列に変換します。

- 引数

  - **x** (`bytes`)：変換する Base64 バイト文字列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。

- **返り値**

  - **np.ndarray**：変換された NumPy 配列。

- **例**

  ```python
  import docsaidkit as D

  b64 = D.npy_to_b64(np.random.rand(100, 100, 3))
  x = D.b64_to_npy(b64)
  ```

## b64str_to_npy

> [b64str_to_npy(x: bytes, dtype='float32', string_encode: str = 'utf-8') -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L170)

- **説明**：Base64 文字列を NumPy 配列に変換します。

- 引数

  - **x** (`bytes`)：変換する Base64 文字列。
  - **dtype** (`str`)：データ型。デフォルトは `'float32'`。
  - **string_encode** (`str`)：文字列エンコード。デフォルトは `'utf-8'`。

- **返り値**

  - **np.ndarray**：変換された NumPy 配列。

- **例**

  ```python
  import docsaidkit as D

  b64 = D.npy_to_b64(np.random.rand(100, 100, 3))
  x = D.b64str_to_npy(b64.decode('utf-8'))
  ```
