---
sidebar_position: 2
---

# Timer

> [Timer(precision: int = 5, desc: str = None, verbose: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L76C1-L157C71)

- **説明**：これはプログラムの実行時間を測定するためのタイマーです。このタイマーは 3 つの使用方法があります：1. `tic` と `toc` メソッドを使用する方法；2. デコレーターを使用する方法；3. `with` 文を使用する方法。ちなみに、このタイマーの設計時、`start/stop` か `tic/toc` という名前で悩みましたが、最終的に `tic/toc` の方がタイマーっぽいと感じて選びました。

- **パラメータ**

  - **precision** (`int`)：小数点の精度。デフォルトは 5。
  - **desc** (`str`)：説明文字。デフォルトは None。
  - **verbose** (`bool`)：タイマー結果を表示するかどうか。デフォルトは False。

- **メソッド**

  - **tic()**：計測を開始します。
  - **toc(verbose=False)**：計測を終了し、経過時間を返します。
  - **clear_record()**：記録をクリアします。

- **属性**

  - **mean** (`float`)：平均時間。
  - **max** (`float`)：最大時間。
  - **min** (`float`)：最小時間。
  - **std** (`float`)：標準偏差。

- **例**

  ```python
  import docsaidkit as D
  import time

  # 'tic' と 'toc' メソッドを使用
  t = D.Timer()
  t.tic()
  time.sleep(1)
  t.toc()

  # デコレーターを使用
  @D.Timer()
  def testing_function():
      time.sleep(1)

  # 'with' 文を使用
  with D.Timer():
      time.sleep(1)
  ```
