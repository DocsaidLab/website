---
sidebar_position: 2
---

# get_curdir

> [get_curdir() -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_path.py#L8)

- **説明**：現在の作業ディレクトリのパスを取得します。ここでの作業ディレクトリとは「この関数を呼び出した」時点での Python ファイルが存在するディレクトリのことです。通常、この方法は相対パスの基準として使用されます。

- **戻り値**

  - **str**：現在の作業ディレクトリのパス。

- **例**

  ```python
  import docsaidkit as D

  DIR = D.get_curdir()
  print(DIR)
  # >>> '/path/to/your/current/directory'
  ```
