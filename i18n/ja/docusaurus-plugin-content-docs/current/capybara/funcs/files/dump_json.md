# dump_json

> [dump_json(obj: Any, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L65)

- **説明**：オブジェクトを json 形式で書き込みます。ここでは `ujson` を使って書き込んでおり、その理由は `ujson` の方が `json` よりもかなり速いためです。

- **引数**

  - **obj** (`Any`)：書き込む対象のオブジェクト。
  - **path** (`Union[str, Path]`)：json ファイルのパス。デフォルトは None で、現在のディレクトリに `tmp.json` という名前で書き込まれます。
  - `**kwargs`：`ujson.dump` のその他の引数。

- **使用例**

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_json(data)
  ```
