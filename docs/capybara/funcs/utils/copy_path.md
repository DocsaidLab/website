# copy_path

> [copy_path(path_src: str | Path, path_dst: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **說明**：複製檔案。

- **參數**

  - **path_src** (`str | Path`)：來源檔案路徑（必須是檔案）。
  - **path_dst** (`str | Path`)：目標路徑。

- **例外**

  - **ValueError**：`path_src` 不是檔案時會拋出。

- **範例**

  ```python
  from capybara.utils import copy_path

  path_src = '/path/to/your/source'
  path_dst = '/path/to/your/destination'
  copy_path(path_src, path_dst)
  ```
