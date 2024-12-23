# load_json

> [load_json(path: Union[Path, str], \*\*kwargs) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L50)

- **説明**：json ファイルを読み込む。この処理は `ujson` を使って行います。理由は `ujson` が `json` よりもはるかに速いためです。

- **パラメータ**

  - **path** (`Union[Path, str]`)：json ファイルのパス。
  - `**kwargs`：`ujson.load` のその他のパラメータ。

- **戻り値**

  - **dict**：json ファイルの内容。

- **例**

  ```python
  import capybara as cb

  path = '/path/to/your/json'
  data = cb.load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
