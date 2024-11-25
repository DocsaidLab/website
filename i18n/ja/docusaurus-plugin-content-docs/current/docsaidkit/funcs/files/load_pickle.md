---
sidebar_position: 5
---

# load_pickle

> [load_pickle(path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L159)

- **説明**：pickle ファイルを読み込みます。

- **パラメータ**

  - **path** (`Union[str, Path]`)：pickle ファイルのパス。

- **戻り値**

  - **dict**：pickle ファイルの内容。

- **例**

  ```python
  import docsaidkit as D

  path = '/path/to/your/pickle'
  data = D.load_pickle(path)
  print(data)
  # >>> {'key': 'value'}
  ```
