---
sidebar_position: 3
---

# load_json

> [load_json(path: Union[Path, str], **kwargs) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L50)

- **說明**：讀取 json 檔案。這裡是透過 `ujson` 來讀取，原因是 `ujson` 比 `json` 快很多。

- **參數**
    - **path** (`Union[Path, str]`)：json 檔案的路徑。
    - `**kwargs`：`ujson.load` 的其他參數。

- **傳回值**
    - **dict**：json 檔案的內容。

- **範例**

    ```python
    import docsaidkit as D

    path = '/path/to/your/json'
    data = D.load_json(path)
    print(data)
    # >>> {'key': 'value'}
    ```
