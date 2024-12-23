# Timer

> [Timer(precision: int = 5, desc: str = None, verbose: bool = False)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/time.py#L76)

- **説明**：これは、プログラムの実行時間を計測するためのタイマーです。このタイマーには 3 つの使い方があります：1. `tic` と `toc` メソッドを使用；2. デコレータを使用；3. `with` ステートメントを使用。設計時に `start/stop` と `tic/toc` どちらにするか迷いましたが、計測を意識して `tic/toc` を選びました。

- **パラメータ**

  - **precision** (`int`)：小数点の精度。デフォルトは 5。
  - **desc** (`str`)：説明文。デフォルトは None。
  - **verbose** (`bool`)：計測結果を表示するかどうか。デフォルトは False。

- **メソッド**

  - **tic()**：計測開始。
  - **toc(verbose=False)**：計測終了し、経過時間を返す。
  - **clear_record()**：記録をクリアする。

- **属性**

  - **mean** (`float`)：平均時間。
  - **max** (`float`)：最大時間。
  - **min** (`float`)：最小時間。
  - **std** (`float`)：標準偏差。

- **例**

  ```python
  import capybara as cb

  # 'tic' と 'toc' メソッドを使用
  t = cb.Timer()
  t.tic()
  time.sleep(1)
  t.toc()

  # デコレータを使用
  @cb.Timer()
  def testing_function():
      time.sleep(1)

  # 'with' ステートメントを使用
  with cb.Timer():
      time.sleep(1)
  ```
