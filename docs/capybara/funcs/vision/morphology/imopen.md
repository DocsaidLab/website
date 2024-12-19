---
sidebar_position: 3
---

# imopen

>[imopen(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L76C5-L76C11)


- **說明**：開運算：先侵蝕再膨脹的過程，可以用來消除小物體、斷開物體、平滑物體的邊緣、消除小孔洞等。對於多通道圖像，每個通道都將獨立處理。

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
                    [0, 0, 1, 1, 0], # <- Look at this row
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=np.uint8)

    opened_img = D.imopen(img, ksize=3, kstruct='RECT')

    # Kernel will be like this:
    # >>> np.array([[1, 1, 1],
    #               [1, 1, 1],
    #               [1, 1, 1]], dtype=np.uint8)

    # After opening, the image will be like this:
    # >>> np.array([[1, 1, 1, 0, 0],
    #               [1, 1, 1, 0, 0],
    #               [1, 1, 1, 0, 0],
    #               [0, 0, 0, 0, 0], # <- 1's are removed
    #               [0, 0, 0, 1, 1],
    #               [0, 0, 0, 1, 1]], dtype=np.uint8)
    ```



