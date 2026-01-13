# dump_json

> [dump_json(obj: Any, path: str | Path | None = None, **kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **說明**：將物件寫入 json。這裡是透過 `ujson` 來寫入，原因是 `ujson` 比 `json` 快很多。

- **參數**

  - **obj** (`Any`)：要寫入的物件。
  - **path** (`Union[str, Path]`)：json 檔案的路徑。預設為 None，表示寫入到當前目錄下的 `tmp.json`。
  - `**kwargs`：`ujson.dump` 的其他參數。

- **範例**

  ```python
  from capybara.utils.files_utils import dump_json

  data = {'key': 'value'}
  dump_json(data)
  ```
