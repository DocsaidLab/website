# now

> [now(typ: str = 'timestamp', fmt: str = None) -> Union[float, datetime, time.struct_time]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L161)

- **Description**: Retrieves the current time. You can specify the output type of the time or provide a formatting rule to obtain the time as a string, e.g., `now(fmt='%Y-%m-%d')`.

- **Parameters**

  - **typ** (`str`): Specifies the output type of the time. Supported types are `{'timestamp', 'datetime', 'time'}`. The default is `'timestamp'`.
  - **fmt** (`str`): Specifies the formatting rule for the time. The default is `None`.

- **Example**

  ```python
  import capybara as cb

  # Get now time with timestamp type
  now_time = cb.now()
  print(now_time)
  # >>> 1632214400.0

  # Get now time with datetime type
  now_time = cb.now(typ='datetime')
  print(now_time)
  # >>> 2021-09-22 00:00:00

  # Get now time with time type
  now_time = cb.now(typ='time')
  print(now_time)
  # >>> time.struct_time(tm_year=2021, tm_mon=9, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=265, tm_isdst=0)

  # Get now time with formatted rule
  now_time = cb.now(fmt='%Y-%m-%d')
  print(now_time)
  # >>> '2021-09-22'
  ```
