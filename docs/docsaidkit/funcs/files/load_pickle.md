---
sidebar_position: 5
---

# load_pickle

> [load_pickle(path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L159)

- **說明**：讀取 pickle 檔案。

- **參數**
    - **path** (`Union[str, Path]`)：pickle 檔案的路徑。

- **傳回值**
    - **dict**：pickle 檔案的內容。

- **範例**

    ```python
    import docsaidkit as D

    path = '/path/to/your/pickle'
    data = D.load_pickle(path)
    print(data)
    # >>> {'key': 'value'}
    ```
