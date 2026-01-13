# load_yaml

> [load_yaml(path: Path | str) -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **説明**：yaml ファイルを読み込む。

- **パラメータ**

  - **path** (`Union[Path, str]`)：yaml ファイルのパス。

- **戻り値**

  - **dict**：yaml ファイルの内容。

- **例**

  ```python
  from capybara.utils.files_utils import load_yaml

  path = '/path/to/your/yaml'
  data = load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
