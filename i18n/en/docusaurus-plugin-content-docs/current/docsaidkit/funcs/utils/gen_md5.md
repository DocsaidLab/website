---
sidebar_position: 7
---

# gen_md5

> [gen_md5(file: Union[str, Path], block_size: int = 256 * 128) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L21)

- **Description**: Generates an MD5 hash based on the given file. This function is designed to facilitate accessing a large number of files without the need for manual naming. MD5 hashing is used for this purpose.

- **Parameters**:
    - **file** (`Union[str, Path]`): The file name or path.
    - **block_size** (`int`): The size of each block read. Default is 256*128.

- **Returns**:
    - **str**: The MD5 hash.

- **Example**:

    ```python
    import docsaidkit as D

    file = '/path/to/your/file'
    md5 = D.gen_md5(file)
    print(md5)
    # >>> 'd41d8cd98f00b204e9800998ecf8427e'
    ```
