# load_json

> [load_json(path: Union[Path, str], \*\*kwargs) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L50)

- **說明**：讀取 json 檔案。這裡是透過 `ujson` 來讀取，原因是 `ujson` 比 `json` 快很多。

- **參數**

  - **path** (`Union[Path, str]`)：json 檔案的路徑。
  - `**kwargs`：`ujson.load` 的其他參數。

- **傳回值**

  - **dict**：json 檔案的內容。

- **範例**

  ```python
  import capybara as cb

  path = '/path/to/your/json'
  data = cb.load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
