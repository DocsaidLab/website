---
sidebar_position: 5
---

# PNG Process

## pngencode

> [pngencode(img: np.ndarray, compression: int = 1) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L80)

- **說明**：將 NumPy 圖像數組編碼為 PNG 格式的字節串。

- **參數**：
  - **img** (`np.ndarray`)：要編碼的圖像數組。
  - **compression** (`int`)：壓縮級別，範圍為 0 至 9。0 表示無壓縮，9 表示最高壓縮。預設為 1。

- **傳回值**
    - **bytes**：編碼後的 PNG 格式字節串。

- **範例**

    ```python
    import numpy as np
    import docsaidkit as D

    img_array = np.random.rand(100, 100, 3) * 255
    encoded_bytes = D.pngencode(img_array, compression=9)
    ```

## pngdecode

> [pngdecode(byte_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L91)

- **說明**：將 PNG 格式的字節串解碼為 NumPy 圖像數組。

- **參數**：
  - **byte_** (`bytes`)：要解碼的 PNG 格式的字節串。

- **傳回值**
    - **np.ndarray**：解碼後的圖像數組。

- **範例**

    ```python
    decoded_img = D.pngdecode(encoded_bytes)
    ```


