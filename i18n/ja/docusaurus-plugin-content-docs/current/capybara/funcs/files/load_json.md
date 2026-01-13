# load_json

> [load_json(path: Path | str, **kwargs) -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **説明**：json ファイルを読み込む。この処理は `ujson` を使って行います。理由は `ujson` が `json` よりもはるかに速いためです。

- **パラメータ**

  - **path** (`Union[Path, str]`)：json ファイルのパス。
  - `**kwargs`：`ujson.load` のその他のパラメータ。

- **戻り値**

  - **dict**：json ファイルの内容。

- **例**

  ```python
  from capybara.utils.files_utils import load_json

  path = '/path/to/your/json'
  data = load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
