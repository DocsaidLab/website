# dump_pickle

> [dump_pickle(obj: Any, path: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **說明**：將物件寫入 pickle 檔案。

- **參數**

  - **obj** (`Any`)：要寫入的物件。
  - **path** (`Union[str, Path]`)：pickle 檔案的路徑。

- **範例**

  ```python
  from capybara.utils.files_utils import dump_pickle

  data = {'key': 'value'}
  dump_pickle(data, '/path/to/your/pickle')
  ```
