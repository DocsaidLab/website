# load_yaml

> [load_yaml(path: Union[Path, str]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L185)

- **說明**：讀取 yaml 檔案。

- **參數**

  - **path** (`Union[Path, str]`)：yaml 檔案的路徑。

- **傳回值**

  - **dict**：yaml 檔案的內容。

- **範例**

  ```python
  import capybara as cb

  path = '/path/to/your/yaml'
  data = cb.load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
