---
sidebar_position: 21
---

# download

## gen_download_cmd

> [gen_download_cmd(file_id: str, target: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L68)

- **説明**：Google Drive のファイルをダウンロードするためのコマンドを生成します。

- **パラメータ**

  - **file_id** (`str`)：ファイル ID。
  - **target** (`str`)：ダウンロード先のファイルパス。

- **例**

  ```python
  import docsaidkit as D

  file_id = '1c1b9b1b0cdcwfjowief'
  target = 'example.txt'
  cmd = D.gen_download_cmd(file_id, target)
  print(cmd)
  # >>> wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget
  # >>> --quiet
  # >>> --save-cookies /tmp/cookies.txt
  # >>> --keep-session-cookies
  # >>> --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c1b9b1b0cdcwfjowief'
  # >>> -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c1b9b1b0cdcwfjowief" -O example.txt && rm -rf /tmp/cookies.txt
  ```

## download_from_docsaid

> [download_from_docsaid(file_id: str, file_name: str, target: str) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L79)

- **説明**：Docsaid のプライベートクラウドからデータをダウンロードします。

- **パラメータ**

  - **file_id** (`str`)：ファイル ID。
  - **file_name** (`str`)：ファイル名。
  - **target** (`str`)：ダウンロード先のファイルパス。

- **例**

  ```python
  import docsaidkit as D

  file_id = 'c1b9b1b0cdcwfjowief'
  file_name = 'example.txt'
  target = 'example.txt'
  D.download_from_docsaid(file_id, file_name, target)
  ```
