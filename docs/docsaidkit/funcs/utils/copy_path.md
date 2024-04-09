---
sidebar_position: 5
---

# copy_path

>[copy_path(path_src: Union[str, Path], path_dst: Union[str, Path]) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_path.py#L34)

- **說明**：複製檔案／目錄。

- **參數**
    - **path_src** (`path_src: Union[str, Path]`)：要複製的檔案／目錄。
    - **path_dst** (`path_dst: Union[str, Path]`)：複製後的檔案／目錄。

- **範例**

    ```python
    import docsaidkit as D

    path_src = '/path/to/your/source'
    path_dst = '/path/to/your/destination'
    D.copy_path(path_src, path_dst)
    ```

