---
sidebar_position: 5
---

# copy_path

> [copy_path(path_src: Union[str, Path], path_dst: Union[str, Path]) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_path.py#L34)

- **説明**：ファイルやディレクトリをコピーします。

- **パラメータ**

  - **path_src** (`path_src: Union[str, Path]`)：コピー元のファイルまたはディレクトリ。
  - **path_dst** (`path_dst: Union[str, Path]`)：コピー先のファイルまたはディレクトリ。

- **例**

  ```python
  import docsaidkit as D

  path_src = '/path/to/your/source'
  path_dst = '/path/to/your/destination'
  D.copy_path(path_src, path_dst)
  ```
