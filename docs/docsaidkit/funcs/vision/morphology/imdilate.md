---
sidebar_position: 2
---

# imdilate

>[imdilate(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L45)

- **說明**：膨脹操作：使用指定的結構元素膨脹源圖像，該結構元素確定了取最大值的像素鄰域的形狀。對於多通道圖像，每個通道都將獨立處理。

- **參數**

    - **img** (`np.ndarray`)：輸入圖像。
    - **ksize** (`Union[int, Tuple[int, int]]`)：結構元素的大小。預設為 (3, 3)。
    - **kstruct** (`MORPH`)：元素形狀，可以是 "MORPH.CROSS", "MORPH.RECT", "MORPH.ELLIPSE" 之一。預設為 "MORPH.RECT"。

- **範例**

    ```python
    import numpy as np
    import docsaidkit as D

    img = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]], dtype=np.uint8)

    dilated_img = D.imdilate(img, ksize=3, kstruct='RECT')

    # Kernel will be like this:
    # >>> np.array([[1, 1, 1],
    #               [1, 1, 1],
    #               [1, 1, 1]], dtype=np.uint8)

    # After dilation, the image will be like this:
    # >>> np.array([[1, 1, 1, 1, 1],
    #               [1, 1, 1, 1, 1],
    #               [1, 1, 1, 1, 1],
    #               [1, 1, 1, 1, 1],
    #               [1, 1, 1, 1, 1]], dtype=np.uint8)
    ```

