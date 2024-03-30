---
sidebar_position: 16
---

# PowerDict

>[PowerDict(d=None, **kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/powerdict.py#L10)

- **說明**：這個類別用來建立一個具有凍結和解凍功能的字典，並且可以透過 `.` 的方式來取得內部屬性。

- **屬性**

    - **is_frozen** (`bool`)：判斷字典是否被凍結。

- **方法**

    - **freeze()**：凍結字典。
    - **melt()**：解凍字典。
    - **to_dict()**：將字典轉換為標準字典。
    - **to_json(path: Union[str, Path]) -> None**：將字典寫入 json 檔案。
    - **to_yaml(path: Union[str, Path]) -> None**：將字典寫入 yaml 檔案。
    - **to_txt(path: Union[str, Path]) -> None**：將字典寫入 txt 檔案。
    - **to_pickle(path: Union[str, Path]) -> None**：將字典寫入 pickle 檔案。

－ **類別方法**

    - **load_json(path: Union[str, Path]) -> PowerDict**：從 json 檔案讀取字典。
    - **load_pickle(path: Union[str, Path]) -> PowerDict**：從 pickle 檔案讀取字典。
    - **load_yaml(path: Union[str, Path]) -> PowerDict**：從 yaml 檔案讀取字典。

- **參數**
    - **d** (`dict`, optional)：要轉換的字典。預設為 None。

- **範例**

    ```python
    from docsaidkit import PowerDict

    d = {'key': 'value'}
    cfg = PowerDict(d)
    print(cfg.key)
    # >>> 'value'
    ```

