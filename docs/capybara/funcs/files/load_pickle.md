# load_pickle

> [load_pickle(path: str | Path)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **說明**：讀取 pickle 檔案。

- **參數**

  - **path** (`Union[str, Path]`)：pickle 檔案的路徑。

- **傳回值**

  - **Any**：pickle 檔案反序列化後的內容。

- **範例**

  ```python
  from capybara.utils.files_utils import load_pickle

  path = '/path/to/your/pickle'
  data = load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
