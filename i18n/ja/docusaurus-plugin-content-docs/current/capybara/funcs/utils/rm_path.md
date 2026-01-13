# rm_path

> [rm_path(path: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **説明**：指定されたパスまたはファイルを削除します。

- **引数**

  - **path** (`path: Union[str, Path]`)：削除するパスまたはファイル。

- **例**

  ```python
  from capybara.utils import rm_path

  path = '/path/to/your/directory'
  rm_path(path)
  ```
