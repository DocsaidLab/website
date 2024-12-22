# load_pickle

> [load_pickle(path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L159)

- **說明**：讀取 pickle 檔案。

- **參數**

  - **path** (`Union[str, Path]`)：pickle 檔案的路徑。

- **傳回值**

  - **dict**：pickle 檔案的內容。

- **範例**

  ```python
  import capybara as cb

  path = '/path/to/your/pickle'
  data = cb.load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
