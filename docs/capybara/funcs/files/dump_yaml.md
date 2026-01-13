# dump_yaml

> [dump_yaml(obj: Any, path: str | Path | None = None, **kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **說明**：將物件寫入 yaml 檔案。

- **參數**

  - **obj** (`Any`)：要寫入的物件。
  - **path** (`Union[str, Path]`)：yaml 檔案的路徑。預設為 None，表示寫入到當前目錄下的 `tmp.yaml`。
  - `**kwargs`：`yaml.dump` 的其他參數。

- **範例**

  ```python
  from capybara.utils.files_utils import dump_yaml

  data = {'key': 'value'}
  dump_yaml(data)
  ```
