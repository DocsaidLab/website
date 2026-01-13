# copy_path

> [copy_path(path_src: str | Path, path_dst: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **説明**：ファイルやディレクトリをコピーします。

- **引数**

  - **path_src** (`path_src: Union[str, Path]`)：コピーする元のファイルやディレクトリ。
  - **path_dst** (`path_dst: Union[str, Path]`)：コピー先のファイルやディレクトリ。

- **例**

  ```python
  from capybara.utils import copy_path

  path_src = '/path/to/your/source'
  path_dst = '/path/to/your/destination'
  copy_path(path_src, path_dst)
  ```
