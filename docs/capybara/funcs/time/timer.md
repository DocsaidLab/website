# Timer

> [Timer(precision: int = 5, desc: str = None, verbose: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L76)

- **說明**：這是一個計時器，可以用來計算程式執行的時間。這個計時器有三種使用方式：1. 使用 `tic` 和 `toc` 方法；2. 使用裝飾器；3. 使用 `with` 敘述。順帶一提，當初設計時，糾結在要取名為 `start/stop` 還是 `tic/toc`，最後選擇了 `tic/toc`，因為我覺得這樣比較有計時的感覺。

- **參數**

  - **precision** (`int`)：小數點的精度。預設為 5。
  - **desc** (`str`)：描述文字。預設為 None。
  - **verbose** (`bool`)：是否顯示計時結果。預設為 False。

- **方法**

  - **tic()**：開始計時。
  - **toc(verbose=False)**：結束計時，並回傳經過的時間。
  - **clear_record()**：清除記錄。

- **屬性**

  - **mean** (`float`)：平均時間。
  - **max** (`float`)：最大時間。
  - **min** (`float`)：最小時間。
  - **std** (`float`)：標準差。

- **範例**

  ```python
  import capybara as cb

  # Using 'tic' and 'toc' method
  t = cb.Timer()
  t.tic()
  time.sleep(1)
  t.toc()

  # Using decorator
  @cb.Timer()
  def testing_function():
      time.sleep(1)

  # Using 'with' statement
  with cb.Timer():
      time.sleep(1)
  ```
