---
sidebar_position: 14
---

# load_yaml

> [load_yaml(path: Union[Path, str]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L185)

- **說明**：讀取 yaml 檔案。

- **參數**
    - **path** (`Union[Path, str]`)：yaml 檔案的路徑。

- **傳回值**
    - **dict**：yaml 檔案的內容。

- **範例**

    ```python
    import docsaidkit as D

    path = '/path/to/your/yaml'
    data = D.load_yaml(path)
    print(data)
    # >>> {'key': 'value'}
    ```


