# load_pickle

> [load_pickle(path: str | Path)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **説明**：pickle ファイルを読み込む。

- **パラメータ**

  - **path** (`Union[str, Path]`)：pickle ファイルのパス。

- **戻り値**

  - **dict**：pickle ファイルの内容。

- **例**

  ```python
  from capybara.utils.files_utils import load_pickle

  path = '/path/to/your/pickle'
  data = load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
