---
sidebar_position: 1
---

# imread

>[imread(path: Union[str, Path], color_base: str = 'BGR', verbose: bool = False) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L197C1-L242C15)

- **說明**：讀取圖片，基於不同的影像格式，使用不同的讀取方式，其支援的格式說明如下：
  - `.heic`：使用 `read_heic_to_numpy` 讀取，並轉換成 `BGR` 格式。
  - `.jpg`：使用 `jpgread` 讀取，並轉換成 `BGR` 格式。
  - 其他格式：使用 `cv2.imread` 讀取，並轉換成 `BGR` 格式。
  - 若使用 `jpgread` 讀取為 `None`，則會使用 `cv2.imread` 進行讀取。

- **參數**

    - **path** (`Union[str, Path]`)：要讀取的圖片路徑。
    - **color_base** (`str`)：圖片的色彩空間。如果不是 `BGR`，將使用 `imcvtcolor` 函數進行轉換。預設為 `BGR`。
    - **verbose** (`bool`)：如果設置為 True，當讀取的圖片為 None 時，將發出警告。預設為 False。

- **傳回值**

    - **np.ndarray**：成功時返回圖片的 numpy ndarray，否則返回 None。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    ```
