---
sidebar_position: 5
---

# draw_text

> [draw_text(img: np.ndarray, text: str, location: np.ndarray, color: tuple = (0, 0, 0), text_size: int = 12, font_path: str = None, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L211)

- **說明**：在影像上指定位置繪製文字。

- **參數**
    - **img** (`np.ndarray`)：要繪製的影像。
    - **text** (`str`)：要繪製的文字。
    - **location** (`np.ndarray`)：文字應繪製的 x, y 座標。
    - **color** (`tuple`)：文字顏色的 RGB 值。預設為黑色 (0, 0, 0)。
    - **text_size** (`int`)：要繪製的文字大小。預設為 12。
    - **font_path** (`str`)：要使用的字型檔案的路徑。如果未提供，則使用預設字型 "NotoSansMonoCJKtc-VF.ttf"。
    - **kwargs**：繪製時的其他參數，取決於所使用的底層函式庫或方法。

- **傳回值**
    - **np.ndarray**：繪製了文字的影像。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    text_img = D.draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
    ```

    ![draw_text](./resource/test_draw_text.jpg)

