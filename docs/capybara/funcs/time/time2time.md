# Time2Time

這是一個用來轉換時間格式的工具。

在 Python 中，多個不同時間套件的轉換一直都是個惱人的問題。

為了解決這個問題，我們開發了幾個轉換函數，使得在 datetime、struct_time、timestamp 和 時間字串 之間的轉換變得自由。

以下是這些函數之間關係圖：

![time2time](time2time.png)

如果你好奇上面那張圖是怎麼畫出來的，可以參考下面的 Mermaid 程式碼：

```mermaid
graph TD
    timestamp(timestamp)
    struct_time(struct_time)
    datetime(datetime)
    str(時間字串)

    timestamp -->|timestamp2datetime| datetime
    timestamp -->|timestamp2time| struct_time
    timestamp -->|timestamp2str| str

    struct_time -->|time2datetime| datetime
    struct_time -->|time2timestamp| timestamp
    struct_time -->|time2str| str

    datetime -->|datetime2time| struct_time
    datetime -->|datetime2timestamp| timestamp
    datetime -->|datetime2str| str

    str -->|str2time| struct_time
    str -->|str2datetime| datetime
    str -->|str2timestamp| timestamp
```

看圖說故事，先找到你需要的轉換函數之後，再往下找：

---

## timestamp2datetime

> [timestamp2datetime(ts: Union[int, float]) -> datetime](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L189)

- **說明**：將時間戳轉換為 `datetime`。

- **參數**

  - **ts** (`Union[int, float]`)：時間戳。

- **傳回值**

  - **datetime**：`datetime`。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  dt = cb.timestamp2datetime(ts)
  print(dt)
  # >>> 2021-10-12 16:00:00
  ```

## timestamp2time

> [timestamp2time(ts: Union[int, float]) -> struct_time](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L193)

- **說明**：將時間戳轉換為 `struct_time`。

- **參數**

  - **ts** (`Union[int, float]`)：時間戳。

- **傳回值**

  - **struct_time**：`struct_time`。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  t = cb.timestamp2time(ts)
  print(t)
  # >>> time.struct_time(tm_year=2021, tm_mon=10, tm_mday=12, tm_hour=16, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=285, tm_isdst=0)
  ```

## timestamp2str

> [timestamp2str(ts: Union[int, float], fmt: str) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L197)

- **說明**：將時間戳轉換為時間字串。

- **參數**

  - **ts** (`Union[int, float]`)：時間戳。
  - **fmt** (`str`)：時間格式。

- **傳回值**

  - **str**：時間字串。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  s = cb.timestamp2str(ts, fmt='%Y-%m-%d %H:%M:%S')
  print(s)
  # >>> '2021-10-12 16:00:00'
  ```

## time2datetime

> [time2datetime(t: struct_time) -> datetime](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L201)

- **說明**：將 `struct_time` 轉換為 `datetime`。

- **參數**

  - **t** (`struct_time`)：`struct_time`。

- **傳回值**

  - **datetime**：`datetime`。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  t = cb.timestamp2time(ts)
  dt = cb.time2datetime(t)
  print(dt)
  # >>> datetime.datetime(2021, 10, 12, 16, 0)
  ```

## time2timestamp

> [time2timestamp(t: struct_time) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L207)

- **說明**：將 `struct_time` 轉換為時間戳。

- **參數**

  - **t** (`struct_time`)：`struct_time`。

- **傳回值**

  - **float**：時間戳。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  t = cb.timestamp2time(ts)
  ts = cb.time2timestamp(t)
  print(ts)
  # >>> 1634025600.0
  ```

## time2str

> [time2str(t: struct_time, fmt: str) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L213)

- **說明**：將 `struct_time` 轉換為時間字串。

- **參數**

  - **t** (`struct_time`)：`struct_time`。
  - **fmt** (`str`)：時間格式。

- **傳回值**

  - **str**：時間字串。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  t = cb.timestamp2time(ts)
  s = cb.time2str(t, fmt='%Y-%m-%d %H:%M:%S')
  print(s)
  # >>> '2021-10-12 16:00:00'
  ```

## datetime2time

> [datetime2time(dt: datetime) -> struct_time](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L219)

- **說明**：將 `datetime` 轉換為 `struct_time`。

- **參數**

  - **dt** (`datetime`)：`datetime`。

- **傳回值**

  - **struct_time**：`struct_time`。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  dt = cb.timestamp2datetime(ts)
  t = cb.datetime2time(dt)
  print(t)
  # >>> time.struct_time(tm_year=2021, tm_mon=10, tm_mday=12, tm_hour=16, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=285, tm_isdst=-1)
  ```

## datetime2timestamp

> [datetime2timestamp(dt: datetime) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L225)

- **說明**：將 `datetime` 轉換為時間戳。

- **參數**

  - **dt** (`datetime`)：`datetime`。

- **傳回值**

  - **float**：時間戳。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  dt = cb.timestamp2datetime(ts)
  ts = cb.datetime2timestamp(dt)
  print(ts)
  # >>> 1634025600.0
  ```

## datetime2str

> [datetime2str(dt: datetime, fmt: str) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L231)

- **說明**：將 `datetime` 轉換為時間字串。

- **參數**

  - **dt** (`datetime`)：`datetime`。
  - **fmt** (`str`)：時間格式。

- **傳回值**

  - **str**：時間字串。

- **範例**

  ```python
  import capybara as cb

  ts = 1634025600
  dt = cb.timestamp2datetime(ts)
  s = cb.datetime2str(dt, fmt='%Y-%m-%d %H:%M:%S')
  print(s)
  # >>> '2021-10-12 16:00:00'
  ```

## str2time

> [str2time(s: str, fmt: str) -> struct_time](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L237)

- **說明**：將時間字串轉換為 `struct_time`。

- **參數**

  - **s** (`str`)：時間字串。
  - **fmt** (`str`)：時間格式。

- **傳回值**

  - **struct_time**：`struct_time`。

- **範例**

  ```python
  import capybara as cb

  s = '2021-10-12 16:00:00'
  t = cb.str2time(s, fmt='%Y-%m-%d %H:%M:%S')
  print(t)
  # >>> time.struct_time(tm_year=2021, tm_mon=10, tm_mday=12, tm_hour=16, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=285, tm_isdst=-1)
  ```

## str2datetime

> [str2datetime(s: str, fmt: str) -> datetime](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L243)

- **說明**：將時間字串轉換為 `datetime`。

- **參數**

  - **s** (`str`)：時間字串。
  - **fmt** (`str`)：時間格式。

- **傳回值**

  - **datetime**：`datetime`。

- **範例**

  ```python
  import capybara as cb

  s = '2021-10-12 16:00:00'
  dt = cb.str2datetime(s, fmt='%Y-%m-%d %H:%M:%S')
  print(dt)
  # >>> datetime.datetime(2021, 10, 12, 16, 0)
  ```

## str2timestamp

> [str2timestamp(s: str, fmt: str) -> float](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L249)

- **說明**：將時間字串轉換為時間戳。

- **參數**

  - **s** (`str`)：時間字串。
  - **fmt** (`str`)：時間格式。

- **傳回值**

  - **float**：時間戳。

- **範例**

  ```python
  import capybara as cb

  s = '2021-10-12 16:00:00'
  ts = cb.str2timestamp(s, fmt='%Y-%m-%d %H:%M:%S')
  print(ts)
  # >>> 1634025600.0
  ```
