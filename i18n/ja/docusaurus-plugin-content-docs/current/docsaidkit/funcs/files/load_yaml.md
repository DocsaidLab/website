---
sidebar_position: 7
---

# load_yaml

> [load_yaml(path: Union[Path, str]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L185)

- **説明**：yaml ファイルを読み込みます。

- **パラメータ**

  - **path** (`Union[Path, str]`)：yaml ファイルのパス。

- **戻り値**

  - **dict**：yaml ファイルの内容。

- **例**

  ```python
  import docsaidkit as D

  path = '/path/to/your/yaml'
  data = D.load_yaml(path)
  print(data)
  # >>> {'key': 'value'}
  ```
