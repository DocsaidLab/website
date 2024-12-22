# dump_json

> [dump_json(obj: Any, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L65)

- **說明**：將物件寫入 json。這裡是透過 `ujson` 來寫入，原因是 `ujson` 比 `json` 快很多。

- **參數**

  - **obj** (`Any`)：要寫入的物件。
  - **path** (`Union[str, Path]`)：json 檔案的路徑。預設為 None，表示寫入到當前目錄下的 `tmp.json`。
  - `**kwargs`：`ujson.dump` 的其他參數。

- **範例**

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_json(data)
  ```
