# dump_pickle

> [dump_pickle(obj, path: Union[str, Path]) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L173)

- **説明**：オブジェクトを pickle ファイルに書き込みます。

- **引数**

  - **obj** (`Any`)：書き込む対象のオブジェクト。
  - **path** (`Union[str, Path]`)：pickle ファイルのパス。

- **使用例**

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_pickle(data, '/path/to/your/pickle')
  ```
