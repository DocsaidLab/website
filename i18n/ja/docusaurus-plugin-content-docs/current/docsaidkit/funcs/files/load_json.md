---
sidebar_position: 3
---

# load_json

> [load_json(path: Union[Path, str], \*\*kwargs) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L50)

- **説明**：json ファイルを読み込みます。ここでは `ujson` を使用して読み込みます。理由は、`ujson` が `json` よりも高速だからです。

- **パラメータ**

  - **path** (`Union[Path, str]`)：json ファイルのパス。
  - `**kwargs`：`ujson.load` のその他のパラメータ。

- **戻り値**

  - **dict**：json ファイルの内容。

- **例**

  ```python
  import docsaidkit as D

  path = '/path/to/your/json'
  data = D.load_json(path)
  print(data)
  # >>> {'key': 'value'}
  ```
