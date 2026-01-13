# dump_json

> [dump_json(obj: Any, path: str | Path | None = None, **kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **説明**：オブジェクトを json 形式で書き込みます。ここでは `ujson` を使って書き込んでおり、その理由は `ujson` の方が `json` よりもかなり速いためです。

- **引数**

  - **obj** (`Any`)：書き込む対象のオブジェクト。
  - **path** (`Union[str, Path]`)：json ファイルのパス。デフォルトは None で、現在のディレクトリに `tmp.json` という名前で書き込まれます。
  - `**kwargs`：`ujson.dump` のその他の引数。

- **使用例**

  ```python
  from capybara.utils.files_utils import dump_json

  data = {'key': 'value'}
  dump_json(data)
  ```
