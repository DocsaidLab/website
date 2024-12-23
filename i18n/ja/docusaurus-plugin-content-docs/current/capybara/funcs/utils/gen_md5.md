# gen_md5

> [gen_md5(file: Union[str, Path], block_size: int = 256 \* 128) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L21)

- **説明**：指定したファイルに基づいて md5 を生成します。この関数の目的は、大量のファイルを効率的に処理するために、ファイル名を付ける手間を省くことです。そこで、md5 を使用します。

- **引数**

  - **file** (`Union[str, Path]`)：ファイル名。
  - **block_size** (`int`)：一度に読み込むサイズ。デフォルトは 256 \* 128。

- **戻り値**

  - **str**：md5 値。

- **例**

  ```python
  import capybara as cb

  file = '/path/to/your/file'
  md5 = cb.gen_md5(file)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
