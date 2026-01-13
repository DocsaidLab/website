# dump_pickle

> [dump_pickle(obj: Any, path: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **説明**：オブジェクトを pickle ファイルに書き込みます。

- **引数**

  - **obj** (`Any`)：書き込む対象のオブジェクト。
  - **path** (`Union[str, Path]`)：pickle ファイルのパス。

- **使用例**

  ```python
  from capybara.utils.files_utils import dump_pickle

  data = {'key': 'value'}
  dump_pickle(data, '/path/to/your/pickle')
  ```
