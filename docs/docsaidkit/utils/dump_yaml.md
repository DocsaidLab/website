---
sidebar_position: 15
---

# dump_yaml

> [dump_yaml(obj, path: Union[str, Path] = None, **kwargs) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L200)

- **說明**：將物件寫入 yaml 檔案。

- **參數**
    - **obj** (`Any`)：要寫入的物件。
    - **path** (`Union[str, Path]`)：yaml 檔案的路徑。預設為 None，表示寫入到當前目錄下的 `tmp.yaml`。
    - `**kwargs`：`yaml.dump` 的其他參數。

- **範例**

    ```python
    import docsaidkit as D

    data = {'key': 'value'}
    D.dump_yaml(data)
    ```

