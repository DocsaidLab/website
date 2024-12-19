---
sidebar_position: 1
---

# get_files

>[get_files(folder: Union[str, Path], suffix: Union[str, List[str], Tuple[str]] = None, recursive: bool = True, return_pathlib: bool = True, sort_path: bool = True, ignore_letter_case: bool = True) -> List[Path]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L88)

- **Description**

    Get all files in the specified folder. Note that the `suffix` here is case-insensitive, but you must remember to include the `.`. Many times, not finding files is due to this issue.

- **Parameters**
    - **folder** (`folder: Union[str, Path]`): The specified folder.
    - **suffix** (`suffix: Union[str, List[str], Tuple[str]]`): The suffix(es) of the files to retrieve. For example: ['.jpg', '.png']. Defaults to None, indicating retrieving all files in the folder.
    - **recursive** (`bool`): Whether to include files in subfolders. Defaults to True.
    - **return_pathlib** (`bool`): Whether to return Path objects. Defaults to True.
    - **sort_path** (`bool`): Whether to return a list of paths sorted in natural order. Defaults to True.
    - **ignore_letter_case** (`bool`): Whether to include suffixes with different cases. Defaults to True.

- **Returns**

    - **List[Path]**: A list of file paths.

- **Example**

    ```python
    import docsaidkit as D

    folder = '/path/to/your/folder'
    files = D.get_files(folder, suffix=['.jpg', '.png'])
    print(files)
    # >>> ['/path/to/your/folder/1.jpg', '/path/to/your/folder/2.png']
    ```
