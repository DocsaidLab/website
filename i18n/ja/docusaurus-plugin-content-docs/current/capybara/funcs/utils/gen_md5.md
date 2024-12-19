---
sidebar_position: 7
---

# gen_md5

> [gen_md5(file: Union[str, Path], block_size: int = 256 \* 128) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/files_utils.py#L21)

- **説明**：指定されたファイルから MD5 ハッシュを生成します。この関数は大量のファイルを処理するために設計されており、ファイル名を決定するのが面倒なため、MD5 を使って一意に識別します。

- **パラメータ**

  - **file** (`Union[str, Path]`)：ファイル名。
  - **block_size** (`int`)：一度に読み取るサイズ。デフォルトは 256\*128。

- **戻り値**

  - **str**：MD5 ハッシュ。

- **例**

  ```python
  import docsaidkit as D

  file = '/path/to/your/file'
  md5 = D.gen_md5(file)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
