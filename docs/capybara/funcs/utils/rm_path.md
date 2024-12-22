# rm_path

> [rm_path(path: Union[str, Path]) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/custom_path.py#L26)

- **說明**：移除路徑／檔案。

- **參數**

  - **path** (`path: Union[str, Path]`)：要移除的路徑／檔案。

- **範例**

  ```python
  import capybara as cb

  path = '/path/to/your/directory'
  new_path = cb.rm_path(path)
  ```
