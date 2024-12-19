---
sidebar_position: 9
---

# npyread

> [npyread(path: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L174)

- **說明**：從 NumPy `.npy` 文件中讀取圖像數組。

- **參數**
    - **path** (`Union[str, Path]`)：`.npy` 文件的路徑。

- **傳回值**
    - **np.ndarray**：讀取的圖像數組。如果讀取失敗，則返回 `None`。

- **範例**

    ```python
    import docsaidkit as D

    img = D.npyread('lena.npy')
    ```

