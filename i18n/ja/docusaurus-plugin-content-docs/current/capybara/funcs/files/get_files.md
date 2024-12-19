---
sidebar_position: 1
---

# get_files

> [get_files(folder: Union[str, Path], suffix: Union[str, List[str], Tuple[str]] = None, recursive: bool = True, return_pathlib: bool = True, sort_path: bool = True, ignore_letter_case: bool = True) -> List[Path]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L88)

- **説明**：指定されたフォルダ内のすべてのファイルを取得します。ここでの `suffix` は大文字と小文字を区別しませんが、必ず `.` を含める必要があります。多くの場合、ファイルが見つからないのはこの理由によるものです。

- **パラメータ**

  - **folder** (`folder: Union[str, Path]`)：指定されたフォルダ。
  - **suffix** (`suffix: Union[str, List[str], Tuple[str]]`)：取得するファイルの拡張子。例えば：['.jpg', '.png']。デフォルトは None、すべてのファイルを取得します。
  - **recursive** (`bool`)：サブフォルダ内のファイルも含むかどうか。デフォルトは True。
  - **return_pathlib** (`bool`)：Path オブジェクトを返すかどうか。デフォルトは True。
  - **sort_path** (`bool`)：パスを自然順でソートするかどうか。デフォルトは True。
  - **ignore_letter_case** (`bool`)：大文字と小文字を無視して拡張子を取得するかどうか。デフォルトは True。

- **戻り値**

  - **List[Path]**：ファイルのパスリスト。

- **例**

  ```python
  import docsaidkit as D

  folder = '/path/to/your/folder'
  files = D.get_files(folder, suffix=['.jpg', '.png'])
  print(files)
  # >>> ['/path/to/your/folder/1.jpg', '/path/to/your/folder/2.png']
  ```
