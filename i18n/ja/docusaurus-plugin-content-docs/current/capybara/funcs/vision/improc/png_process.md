---
sidebar_position: 5
---

# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L80)

- **説明**：NumPy 画像配列を PNG 形式のバイト列にエンコードします。

- 引数：

  - **img** (`np.ndarray`)：エンコードする画像配列。
  - **compression** (`int`)：圧縮レベル。0 から 9 の範囲で指定します。0 は圧縮なし、9 は最高圧縮です。デフォルトは 1。

- **返り値**

  - **bytes**：エンコードされた PNG 形式のバイト列。

- **例**

  ```python
  import numpy as np
  import docsaidkit as D

  img_array = np.random.rand(100, 100, 3) * 255
  encoded_bytes = D.pngencode(img_array, compression=9)
  ```

## pngdecode

> [pngdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L91)

- **説明**：PNG 形式のバイト列を NumPy 画像配列にデコードします。

- 引数：

  - **byte\_** (`bytes`)：デコードする PNG 形式のバイト列。

- **返り値**

  - **np.ndarray**：デコード後の画像配列。

- **例**

  ```python
  decoded_img = D.pngdecode(encoded_bytes)
  ```
