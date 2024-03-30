---
sidebar_position: 18
---

# now

> [now(typ: str = 'timestamp', fmt: str = None) -> Union[float, datetime, time.struct_time]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L160)

- **說明**：取得現在時間。可以指定時間的輸出類型，或是給定格式化規則來取得時間字串，例如：`now(fmt='%Y-%m-%d')`。

- **參數**
    - **typ** (`str`, optional)：指定時間的輸出類型。支援的類型有：`{'timestamp', 'datetime', 'time'}`。預設為 `'timestamp'`。
    - **fmt** (`str`, optional)：指定時間的格式化規則。預設為 `None`。

- **範例**

    ```python
    import docsaidkit as D

    # Get now time with timestamp type
    now_time = D.now()
    print(now_time)
    # >>> 1632214400.0

    # Get now time with datetime type
    now_time = D.now(typ='datetime')
    print(now_time)
    # >>> 2021-09-22 00:00:00

    # Get now time with time type
    now_time = D.now(typ='time')
    print(now_time)
    # >>> time.struct_time(tm_year=2021, tm_mon=9, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=265, tm_isdst=0)

    # Get now time with formatted rule
    now_time = D.now(fmt='%Y-%m-%d')
    print(now_time)
    # >>> '2021-09-22'
    ```

