# get_files

> [get_files(folder: Union[str, Path], suffix: Union[str, List[str], Tuple[str]] = None, recursive: bool = True, return_pathlib: bool = True, sort_path: bool = True, ignore_letter_case: bool = True) -> List[Path]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L88)

- **Description**: Retrieves all files in the specified folder. Note that the `suffix` is case-insensitive, but be sure to include the `.` as part of the suffix. Many times, files are not found due to this issue.

- **Parameters**:

  - **folder** (`Union[str, Path]`): The folder to search for files in.
  - **suffix** (`Union[str, List[str], Tuple[str]]`): The file extensions to retrieve. For example: `['.jpg', '.png']`. Defaults to None, meaning all files in the folder are retrieved.
  - **recursive** (`bool`): Whether to include files in subfolders. Defaults to True.
  - **return_pathlib** (`bool`): Whether to return Path objects. Defaults to True.
  - **sort_path** (`bool`): Whether to return the list of paths sorted naturally. Defaults to True.
  - **ignore_letter_case** (`bool`): Whether to match file extensions case-insensitively. Defaults to True.

- **Return Value**:

  - **List[Path]**: A list of file paths.

- **Example**:

  ```python
  import capybara as cb

  folder = '/path/to/your/folder'
  files = cb.get_files(folder, suffix=['.jpg', '.png'])
  print(files)
  # >>> ['/path/to/your/folder/1.jpg', '/path/to/your/folder/2.png']
  ```
