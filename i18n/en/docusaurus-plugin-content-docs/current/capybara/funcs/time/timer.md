---
sidebar_position: 2
---

# Timer

> [Timer(precision: int = 5, desc: str = None, verbose: bool = False)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L76C1-L157C71)

- **Description**: This is a timer that can be used to measure the execution time of a program. The timer has three usage modes:

    1. using the `tic` and `toc` methods;
    2. using decorators;
    3. using the `with` statement.

- **Parameters**:
    - **precision** (`int`): The precision of decimal points. Default is 5.
    - **desc** (`str`): Description text. Default is None.
    - **verbose** (`bool`): Whether to display the timing results. Default is False.

- **Methods**:
    - **tic()**: Start the timer.
    - **toc(verbose=False)**: Stop the timer and return the elapsed time.
    - **clear_record()**: Clear the records.

- **Attributes**:
    - **mean** (`float`): Mean time.
    - **max** (`float`): Maximum time.
    - **min** (`float`): Minimum time.
    - **std** (`float`): Standard deviation.

- **Example**:

    ```python
    import docsaidkit as D

    # Using 'tic' and 'toc' method
    t = D.Timer()
    t.tic()
    time.sleep(1)
    t.toc()

    # Using decorator
    @D.Timer()
    def testing_function():
        time.sleep(1)

    # Using 'with' statement
    with D.Timer():
        time.sleep(1)
    ```
