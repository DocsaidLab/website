---
sidebar_position: 8
---

# dump_yaml

> [dump_yaml(obj, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L200)

- **説明**：オブジェクトを yaml ファイルに書き出します。

- **パラメータ**

  - **obj** (`Any`)：書き出すオブジェクト。
  - **path** (`Union[str, Path]`)：yaml ファイルのパス。デフォルトは None、現在のディレクトリに `tmp.yaml` として書き出します。
  - `**kwargs`：`yaml.dump` のその他のパラメータ。

- **例**

  ```python
  import docsaidkit as D

  data = {'key': 'value'}
  D.dump_yaml(data)
  ```
