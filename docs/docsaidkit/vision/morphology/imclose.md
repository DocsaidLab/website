---
sidebar_position: 4
---

# imclose

>[imclose(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L105)

- **說明**：閉運算：先膨脹再侵蝕的過程，可以用來填充物體內部的小洞、平滑物體的邊緣、連接兩個物體等。對於多通道圖像，每個通道都將獨立處理。

- **參數**

    - **img** (`np.ndarray`)：輸入圖像。
    - **ksize** (`Union[int, Tuple[int, int]]`)：結構元素的大小。預設為 (3, 3)。
    - **kstruct** (`MORPH`)：元素形狀，可以是 "MORPH.CROSS", "MORPH.RECT", "MORPH.ELLIPSE" 之一。預設為 "MORPH.RECT"。

- **範例**

    ```python
    import numpy as np
    import docsaidkit as D

    img = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0], # <- Look at this row
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=np.uint8)

    closed_img = D.imclose(img, ksize=3, kstruct='CROSS')

    # Kernel will be like this:
    # >>> np.array([[0, 1, 0],
    #               [1, 1, 1],
    #               [0, 1, 0]], dtype=np.uint8)

    # After closing, the image will be like this:
    # >>> np.array([[1, 1, 1, 0, 0],
    #               [1, 1, 1, 0, 0],
    #               [1, 1, 1, 0, 0],
    #               [0, 0, 1, 1, 0], # <- 1's are connected
    #               [0, 0, 0, 1, 1],
    #               [0, 0, 0, 1, 1]], dtype=np.uint8)
    ```
