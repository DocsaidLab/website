---
sidebar_position: 7
---

# imencode

> [imencode(img: np.ndarray, IMGTYP: Union[str, int, IMGTYP] = IMGTYP.JPEG) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L100)

- **說明**：將 NumPy 圖像數組編碼為指定格式的字節串。

- **參數**
    - **img** (`np.ndarray`)：要編碼的圖像數組。
    - **IMGTYP** (`Union[str, int, IMGTYP]`)：圖像類型。支持的類型有 `IMGTYP.JPEG` 和 `IMGTYP.PNG`。預設為 `IMGTYP.JPEG`。

- **傳回值**
    - **bytes**：編碼後的圖像字節串。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    encoded_bytes = D.imencode(img, IMGTYP=D.IMGTYP.PNG)
    ```

