---
sidebar_position: 8
---

# img_to_md5

> [img_to_md5(img: np.ndarray) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L42)

- **說明**：根據給定的影像生成 md5。跟 `gen_md5` 的存在是同一個理由，不同的是這個函數是針對影像輸入的。

- **參數**
    - **img** (`np.ndarray`)：影像。

- **傳回值**

    - **str**：md5。

- **範例**

    ```python
    import docsaidkit as D

    img = D.imread('lena.png')
    md5 = D.img_to_md5(img)
    print(md5)
    # >>> 'd41d8cd98f00b204e9800998ecf8427e'
    ```

