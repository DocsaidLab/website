---
sidebar_position: 6
---

# dump_pickle

> [dump_pickle(obj, path: Union[str, Path]) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L173)

- **説明**：オブジェクトを pickle ファイルに書き出します。

- **パラメータ**

  - **obj** (`Any`)：書き出すオブジェクト。
  - **path** (`Union[str, Path]`)：pickle ファイルのパス。

- **例**

  ```python
  import docsaidkit as D

  data = {'key': 'value'}
  D.dump_pickle(data, '/path/to/your/pickle')
  ```
