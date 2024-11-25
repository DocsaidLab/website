---
sidebar_position: 1
---

# now

> [now(typ: str = 'timestamp', fmt: str = None) -> Union[float, datetime, time.struct_time]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L160)

- **説明**：現在の時間を取得します。時間の出力タイプを指定したり、フォーマットルールを指定して時間文字列を取得することができます。例：`now(fmt='%Y-%m-%d')`。

- **パラメータ**

  - **typ** (`str`)：時間の出力タイプを指定します。サポートされているタイプは：`{'timestamp', 'datetime', 'time'}`。デフォルトは `'timestamp'`。
  - **fmt** (`str`)：時間のフォーマットルールを指定します。デフォルトは `None`。

- **例**

  ```python
  import docsaidkit as D

  # タイムスタンプ型で現在の時間を取得
  now_time = D.now()
  print(now_time)
  # >>> 1632214400.0

  # datetime型で現在の時間を取得
  now_time = D.now(typ='datetime')
  print(now_time)
  # >>> 2021-09-22 00:00:00

  # time型で現在の時間を取得
  now_time = D.now(typ='time')
  print(now_time)
  # >>> time.struct_time(tm_year=2021, tm_mon=9, tm_mday=22, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=2, tm_yday=265, tm_isdst=0)

  # フォーマットルールで現在の時間を取得
  now_time = D.now(fmt='%Y-%m-%d')
  print(now_time)
  # >>> '2021-09-22'
  ```
