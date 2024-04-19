---
sidebar_position: 1
---

# now

> [now(typ: str = 'timestamp', fmt: str = None) -> Union[float, datetime, time.struct_time]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L160)

- **Description**

    Get the current time. You can specify the output type of time or provide a formatting rule to get a time string, for example: `now(fmt='%Y-%m-%d')`.

- **Parameters**
    - **typ** (`str`): Specify the output type of time. Supported types are: `{'timestamp', 'datetime', 'time'}`. Default is `'timestamp'`.
    - **fmt** (`str`): Specify the formatting rule of time. Default is `None`.

- **Example**

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
