---
sidebar_position: 7
---

# imblackhat

>[imblackhat(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/morphology.py#L191)

- **説明**：ブラックハット変換：閉じる操作の結果から元の画像を引きます。複数チャンネル画像の場合、各チャンネルは独立して処理されます。この操作は、元の画像より暗い領域（例えば、暗点や小さな構造）を抽出し、大きな暗い領域を除去または軽減するのに使用されます。

- 引数

    - **img** (`np.ndarray`)：入力画像。
    - **ksize** (`Union[int, Tuple[int, int]]`)：構造要素のサイズ。デフォルトは (3, 3)。
    - **kstruct** (`MORPH`)：要素の形状。`"MORPH.CROSS"`, `"MORPH.RECT"`, `"MORPH.ELLIPSE"` のいずれか。デフォルトは `"MORPH.RECT"`。

- **例**

    ```python
    import numpy as np
    import docsaidkit as D

    img = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0], # <- Look at this row
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=np.uint8)

    blackhat_img = D.imblackhat(img, ksize=3, kstruct='CROSS')

    # Kernel will be like this:
    # >>> np.array([[0, 1, 0],
    #               [1, 1, 1],
    #               [0, 1, 0]], dtype=np.uint8)

    # After blackhat, the image will be like this:
    # >>> np.array([[0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0],
    #               [0, 0, 1, 1, 0], # <- 1's are extracted
    #               [0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0]], dtype=np.uint8)
    ```