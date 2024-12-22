# dump_pickle

> [dump_pickle(obj, path: Union[str, Path]) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L173)

- **說明**：將物件寫入 pickle 檔案。

- **參數**

  - **obj** (`Any`)：要寫入的物件。
  - **path** (`Union[str, Path]`)：pickle 檔案的路徑。

- **範例**

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_pickle(data, '/path/to/your/pickle')
  ```
