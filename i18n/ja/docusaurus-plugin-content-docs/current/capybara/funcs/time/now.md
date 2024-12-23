# now

> [now(typ: str = 'timestamp', fmt: str = None) -> Union[float, datetime, time.struct_time]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L161)

- **説明**：現在の時間を取得します。時間の出力タイプを指定するか、フォーマット規則を与えて時間の文字列を取得できます。例えば、`now(fmt='%Y-%m-%d')` のように指定できます。

- **パラメータ**

  - **typ** (`str`)：時間の出力タイプを指定します。サポートされているタイプは `{'timestamp', 'datetime', 'time'}` です。デフォルトは `'timestamp'`。
  - **fmt** (`str`)：時間のフォーマット規則を指定します。デフォルトは `None`。

- **例**

  ```python
  import capybara as cb

  # タイムスタンプタイプで現在の時間を取得
  now_time = cb.now()
  print(now_time)
  # >>> 1632214400.0

  # datetimeタイプで現在の時間を取得
  now_time = cb.now(typ='datetime')
  print(now_time)
  # >>> 2021-09-22 00:00:00

  # timeタイプで現在の時間を取得
  now_time = cb.now(typ='time')
  print(now_time)
  # >>> time.struct_time(tm_year=2021, tm_mon=9, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=265, tm_isdst=0)

  # フォーマット規則で現在の時間を取得
  now_time = cb.now(fmt='%Y-%m-%d')
  print(now_time)
  # >>> '2021-09-22'
  ```
