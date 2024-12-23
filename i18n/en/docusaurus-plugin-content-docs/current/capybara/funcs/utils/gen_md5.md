# gen_md5

> [gen_md5(file: Union[str, Path], block_size: int = 256 \* 128) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L21)

- **Description**: Generates the MD5 hash for a given file. The purpose of this function is to make it easier to handle a large number of files. Naming them all can be tedious, so MD5 is used instead.

- **Parameters**

  - **file** (`Union[str, Path]`): The file name.
  - **block_size** (`int`): The size of each read block. The default is 256 \* 128.

- **Return value**

  - **str**: The MD5 hash.

- **Example**

  ```python
  import capybara as cb

  file = '/path/to/your/file'
  md5 = cb.gen_md5(file)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
