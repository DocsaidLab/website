# rm_path

> [rm_path(path: str | Path) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/custom_path.py)

- **說明**：移除路徑／檔案。

- **參數**

  - **path** (`str | Path`)：要移除的路徑／檔案。

- **行為**

  - 若 `path` 是資料夾且不是 symlink：使用 `shutil.rmtree` 遞迴刪除。
  - 其他情況（一般檔案／symlink）：使用 `Path.unlink()` 刪除。

- **範例**

  ```python
  from capybara.utils import rm_path

  path = '/path/to/your/directory'
  rm_path(path)
  ```
