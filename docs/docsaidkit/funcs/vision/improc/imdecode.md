---
sidebar_position: 8
---

# imdecode

> [imdecode(byte_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L107)

- **說明**：將圖像字節串解碼為 NumPy 圖像數組。

- **參數**
    - **byte_** (`bytes`)：要解碼的圖像字節串。

- **傳回值**
    - **np.ndarray**：解碼後的圖像數組。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    encoded_bytes = D.imencode(img, IMGTYP=D.IMGTYP.PNG)
    decoded_img = D.imdecode(encoded_bytes)
    ```

