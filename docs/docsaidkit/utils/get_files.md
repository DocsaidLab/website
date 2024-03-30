---
sidebar_position: 9
---

# get_files

>[get_files(folder: Union[str, Path], suffix: Union[str, List[str], Tuple[str]] = None, recursive: bool = True, return_pathlib: bool = True, sort_path: bool = True, ignore_letter_case: bool = True) -> List[Path]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L88)

- **說明**：取得指定資料夾中的所有檔案。注意這裡的 `suffix` 對於大小寫不敏感，但是必須要記得加上 `.`，很多時候沒有找到檔案，就是因為這個問題。

- **參數**
    - **folder** (`folder: Union[str, Path]`)：指定的資料夾。
    - **suffix** (`suffix: Union[str, List[str], Tuple[str]]`)：要取得的檔案的副檔名。例如：['.jpg', '.png']。預設為 None，表示取得資料夾中的所有檔案。
    - **recursive** (`bool`)：是否包含子資料夾中的檔案。預設為 True。
    - **return_pathlib** (`bool`)：是否回傳 Path 物件。預設為 True。
    - **sort_path** (`bool`)：是否回傳自然排序的路徑列表。預設為 True。
    - **ignore_letter_case** (`bool`)：是否取得包含大小寫的副檔名。預設為 True。

- **傳回值**

    - **List[Path]**：檔案的路徑列表。

- **範例**

    ```python
    import docsaidkit as D

    folder = '/path/to/your/folder'
    files = D.get_files(folder, suffix=['.jpg', '.png'])
    print(files)
    # >>> ['/path/to/your/folder/1.jpg', '/path/to/your/folder/2.png']
    ```
