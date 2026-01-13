# dump_yaml

> [dump_yaml(obj: Any, path: str | Path | None = None, **kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **説明**：オブジェクトを yaml ファイルに書き込む。

- **引数**

  - **obj** (`Any`)：書き込むオブジェクト。
  - **path** (`Union[str, Path]`)：yaml ファイルのパス。デフォルトは None で、現在のディレクトリに `tmp.yaml` というファイルに書き込むことを意味します。
  - `**kwargs`：`yaml.dump` のその他の引数。

- **例**

  ```python
  from capybara.utils.files_utils import dump_yaml

  data = {'key': 'value'}
  dump_yaml(data)
  ```
