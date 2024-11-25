---
sidebar_position: 4
---

# dump_json

> [dump_json(obj: Any, path: Union[str, Path] = None, \*\*kwargs) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L65)

- **説明**：オブジェクトを json として書き出します。ここでは `ujson` を使用して書き出します。理由は、`ujson` が `json` よりも高速だからです。

- **パラメータ**

  - **obj** (`Any`)：書き出すオブジェクト。
  - **path** (`Union[str, Path]`)：json ファイルのパス。デフォルトは None、現在のディレクトリに `tmp.json` として書き出します。
  - `**kwargs`：`ujson.dump` のその他のパラメータ。

- **例**

  ```python
  import docsaidkit as D

  data = {'key': 'value'}
  D.dump_json(data)
  ```
