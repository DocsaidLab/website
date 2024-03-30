---
sidebar_position: 21
---

# Download

## gen_download_cmd

> [gen_download_cmd(file_id: str, target: str) -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L68)

- **說明**：生成下載 Google Drive 檔案的命令。

- **參數**
    - **file_id** (`str`)：檔案 ID。
    - **target** (`str`)：下載檔案的目標路徑。

- **範例**

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

- **說明**：到 Docsaid 的私有雲下載資料。

- **參數**
    - **file_id** (`str`)：檔案 ID。
    - **file_name** (`str`)：檔案名稱。
    - **target** (`str`)：下載檔案的目標路徑。

- **範例**

    ```python
    import docsaidkit as D

    file_id = 'c1b9b1b0cdcwfjowief'
    file_name = 'example.txt'
    target = 'example.txt'
    D.download_from_docsaid(file_id, file_name, target)
    ```
