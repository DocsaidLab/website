# get_files

> [get_files(folder: Union[str, Path], suffix: Union[str, List[str], Tuple[str]] = None, recursive: bool = True, return_pathlib: bool = True, sort_path: bool = True, ignore_letter_case: bool = True) -> List[Path]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L88)

- **説明**：指定したフォルダ内のすべてのファイルを取得します。ここでの `suffix` は大文字と小文字を区別しませんが、`suffix` に `.` を含める必要があることに注意してください。多くの場合、この問題でファイルが見つからないことがあります。

- **パラメータ**

  - **folder** (`folder: Union[str, Path]`)：指定するフォルダ。
  - **suffix** (`suffix: Union[str, List[str], Tuple[str]]`)：取得するファイルの拡張子。例えば：['.jpg', '.png']。デフォルトは None で、フォルダ内のすべてのファイルを取得します。
  - **recursive** (`bool`)：サブフォルダ内のファイルも含むかどうか。デフォルトは True。
  - **return_pathlib** (`bool`)：Path オブジェクトを返すかどうか。デフォルトは True。
  - **sort_path** (`bool`)：自然順序でパスのリストを返すかどうか。デフォルトは True。
  - **ignore_letter_case** (`bool`)：拡張子の大文字と小文字を区別するかどうか。デフォルトは True。

- **戻り値**

  - **List[Path]**：ファイルのパスのリスト。

- **例**

  ```python
  import capybara as cb

  folder = '/path/to/your/folder'
  files = cb.get_files(folder, suffix=['.jpg', '.png'])
  print(files)
  # >>> ['/path/to/your/folder/1.jpg', '/path/to/your/folder/2.png']
  ```
