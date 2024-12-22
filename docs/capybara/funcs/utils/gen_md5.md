# gen_md5

> [gen_md5(file: Union[str, Path], block_size: int = 256 \* 128) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L21)

- **說明**：根據給定的檔案生成 md5。設計這個函數的目的是為了方便存取大量檔案，要幫他們取名字太累了，所以就用 md5 。

- **參數**

  - **file** (`Union[str, Path]`)：檔案名稱。
  - **block_size** (`int`)：每次讀取的大小。預設為 256\*128。

- **傳回值**

  - **str**：md5。

- **範例**

  ```python
  import capybara as cb

  file = '/path/to/your/file'
  md5 = cb.gen_md5(file)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
