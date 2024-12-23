# copy_path

> [copy_path(path_src: Union[str, Path], path_dst: Union[str, Path]) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/custom_path.py#L34)

- **説明**：ファイルやディレクトリをコピーします。

- **引数**

  - **path_src** (`path_src: Union[str, Path]`)：コピーする元のファイルやディレクトリ。
  - **path_dst** (`path_dst: Union[str, Path]`)：コピー先のファイルやディレクトリ。

- **例**

  ```python
  import capybara as cb

  path_src = '/path/to/your/source'
  path_dst = '/path/to/your/destination'
  cb.copy_path(path_src, path_dst)
  ```
