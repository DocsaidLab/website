# now

> [now(typ: str = "timestamp", fmt: str | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/time.py)

- **Description**: Retrieves the current time. You can specify the output type of the time or provide a formatting rule to obtain the time as a string, e.g., `now(fmt='%Y-%m-%d')`.

- **Parameters**

  - **typ** (`str`): Specifies the output type of the time. Supported types are `{'timestamp', 'datetime', 'time'}`. The default is `'timestamp'`.
  - **fmt** (`str`): Specifies the formatting rule for the time. The default is `None`.

- **Notes**

  - When `fmt` is not `None`, the return value is always the formatted string and `typ` is ignored (current behavior).

- **Example**

  ```python
  from capybara.utils import now

  # Get now time with timestamp type
  now_time = now()
  print(now_time)
  # >>> 1632214400.0

  # Get now time with datetime type
  now_time = now(typ='datetime')
  print(now_time)
  # >>> 2021-09-22 00:00:00

  # Get now time with time type
  now_time = now(typ='time')
  print(now_time)
  # >>> time.struct_time(tm_year=2021, tm_mon=9, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=265, tm_isdst=0)

  # Get now time with formatted rule
  now_time = now(fmt='%Y-%m-%d')
  print(now_time)
  # >>> '2021-09-22'
  ```
