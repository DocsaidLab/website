---
sidebar_position: 6
---

# dump_pickle

> [dump_pickle(obj, path: Union[str, Path]) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L173)

- **說明**：將物件寫入 pickle 檔案。

- **參數**
    - **obj** (`Any`)：要寫入的物件。
    - **path** (`Union[str, Path]`)：pickle 檔案的路徑。

- **範例**

    ```python
    import docsaidkit as D

    data = {'key': 'value'}
    D.dump_pickle(data, '/path/to/your/pickle')
    ```
