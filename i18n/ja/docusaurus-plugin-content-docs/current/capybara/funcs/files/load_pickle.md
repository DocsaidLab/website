# load_pickle

> [load_pickle(path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L159)

- **説明**：pickle ファイルを読み込む。

- **パラメータ**

  - **path** (`Union[str, Path]`)：pickle ファイルのパス。

- **戻り値**

  - **dict**：pickle ファイルの内容。

- **例**

  ```python
  import capybara as cb

  path = '/path/to/your/pickle'
  data = cb.load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
