---
sidebar_position: 4
---

# dump_json

> [dump_json(obj: Any, path: Union[str, Path] = None, **kwargs) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L65)

- **說明**：將物件寫入 json。這裡是透過 `ujson` 來寫入，原因是 `ujson` 比 `json` 快很多。

- **參數**
    - **obj** (`Any`)：要寫入的物件。
    - **path** (`Union[str, Path]`)：json 檔案的路徑。預設為 None，表示寫入到當前目錄下的 `tmp.json`。
    - `**kwargs`：`ujson.dump` 的其他參數。

- **範例**

    ```python
    import docsaidkit as D

    data = {'key': 'value'}
    D.dump_json(data)
    ```
