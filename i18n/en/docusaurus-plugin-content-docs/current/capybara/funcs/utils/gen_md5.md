# gen_md5

> [gen_md5(file: str | Path, block_size: int = 256 * 128) -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/files_utils.py)

- **Description**: Generates the MD5 hash for a given file. The purpose of this function is to make it easier to handle a large number of files. Naming them all can be tedious, so MD5 is used instead.

- **Parameters**

  - **file** (`Union[str, Path]`): The file name.
  - **block_size** (`int`): The size of each read block. The default is 256 \* 128.

- **Return value**

  - **str**: The MD5 hash.

- **Example**

  ```python
  from capybara.utils.files_utils import gen_md5

  file = '/path/to/your/file'
  md5 = gen_md5(file)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
