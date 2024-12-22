# dump_yaml

> [dump_yaml(obj, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L200)

- **說明**：將物件寫入 yaml 檔案。

- **參數**

  - **obj** (`Any`)：要寫入的物件。
  - **path** (`Union[str, Path]`)：yaml 檔案的路徑。預設為 None，表示寫入到當前目錄下的 `tmp.yaml`。
  - `**kwargs`：`yaml.dump` 的其他參數。

- **範例**

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_yaml(data)
  ```
