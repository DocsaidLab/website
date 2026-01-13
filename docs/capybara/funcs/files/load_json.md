# load_json

> [load_json(path: Path | str, **kwargs) -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **說明**：讀取 json 檔案。這裡是透過 `ujson` 來讀取，原因是 `ujson` 比 `json` 快很多。

- **參數**

  - **path** (`Union[Path, str]`)：json 檔案的路徑。
  - `**kwargs`：`ujson.load` 的其他參數。

- **傳回值**

  - **dict**：json 檔案的內容。

- **範例**

  ```python
  from capybara.utils.files_utils import load_json

  path = '/path/to/your/json'
  data = load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
