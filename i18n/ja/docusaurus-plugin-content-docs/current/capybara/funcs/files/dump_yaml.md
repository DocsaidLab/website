# dump_yaml

> [dump_yaml(obj, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L200)

- **説明**：オブジェクトを yaml ファイルに書き込む。

- **引数**

  - **obj** (`Any`)：書き込むオブジェクト。
  - **path** (`Union[str, Path]`)：yaml ファイルのパス。デフォルトは None で、現在のディレクトリに `tmp.yaml` というファイルに書き込むことを意味します。
  - `**kwargs`：`yaml.dump` のその他の引数。

- **例**

  ```python
  import capybara as cb

  data = {'key': 'value'}
  cb.dump_yaml(data)
  ```
