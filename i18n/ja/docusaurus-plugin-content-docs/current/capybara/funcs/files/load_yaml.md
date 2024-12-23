# load_yaml

> [load_yaml(path: Union[Path, str]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L185)

- **説明**：yaml ファイルを読み込む。

- **パラメータ**

  - **path** (`Union[Path, str]`)：yaml ファイルのパス。

- **戻り値**

  - **dict**：yaml ファイルの内容。

- **例**

  ```python
  import capybara as cb

  path = '/path/to/your/yaml'
  data = cb.load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
