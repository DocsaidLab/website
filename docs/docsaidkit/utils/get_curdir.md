---
sidebar_position: 3
---

# get_curdir

>[get_curdir() -> str](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_path.py#L8)

- **說明**：取得當前工作目錄的路徑。這裡的工作目錄是指「調用本函數」的當下，那個 Python 檔案所在的目錄。一般來說，我們會透過這種方式來作為相對路徑的基準。

- **傳回值**
    - **str**：當前工作目錄的路徑。

- **範例**

    ```python
    import docsaidkit as D

    DIR = D.get_curdir()
    print(DIR)
    # >>> '/path/to/your/current/directory'
    ```
