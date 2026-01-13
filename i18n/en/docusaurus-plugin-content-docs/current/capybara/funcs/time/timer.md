# Timer

> [Timer(precision: int = 5, desc: str | None = None, verbose: bool = False)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/time.py)

- **Description**: This is a timer used to measure the execution time of a program. There are three ways to use this timer: 1. Using `tic` and `toc` methods; 2. Using a decorator; 3. Using a `with` statement. By the way, when designing this, I was torn between naming it `start/stop` or `tic/toc`, and ultimately chose `tic/toc` because it feels more fitting for a timer.

- **Parameters**

  - **precision** (`int`): The precision of the decimal point. Default is 5.
  - **desc** (`str`): A description text. Default is None.
  - **verbose** (`bool`): Whether to display the timing results. Default is False.

- **Methods**

  - **tic()**: Start the timer.
  - **toc(verbose=False)**: End the timer and return the elapsed time.
  - **clear_record()**: Clear the recorded data.

- **Attributes**

  - **mean** (`float`): The average time.
  - **max** (`float`): The maximum time.
  - **min** (`float`): The minimum time.
  - **std** (`float`): The standard deviation.

- **Example**

  ```python
  import time
  from capybara.utils import Timer

  # Using 'tic' and 'toc' method
  t = Timer()
  t.tic()
  time.sleep(1)
  t.toc()

  # Using decorator
  @Timer()
  def testing_function():
      time.sleep(1)

  # Using 'with' statement
  with Timer():
      time.sleep(1)
  ```
