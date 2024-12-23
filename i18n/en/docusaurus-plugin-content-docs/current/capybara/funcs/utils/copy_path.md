# copy_path

> [copy_path(path_src: Union[str, Path], path_dst: Union[str, Path]) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/custom_path.py#L34)

- **Description**: Copies a file or directory.

- **Parameters**

  - **path_src** (`Union[str, Path]`): The source file or directory to be copied.
  - **path_dst** (`Union[str, Path]`): The destination where the file or directory will be copied.

- **Example**

  ```python
  import capybara as cb

  path_src = '/path/to/your/source'
  path_dst = '/path/to/your/destination'
  cb.copy_path(path_src, path_dst)
  ```
