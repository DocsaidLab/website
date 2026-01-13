# make_batch

> [make_batch(data: Iterable | Generator, batch_size: int) -> Generator[list, None, None]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/utils.py)

- **Description**: This function is used to convert data into batches.

- **Parameters**

  - **data** (`Union[Iterable, Generator]`): The data generator.
  - **batch_size** (`int`): The size of each batch.

- **Returns**

  - **Generator[List, None, None]**: A generator that yields batched data.

- **Example**

  ```python
  import capybara as cb

  data = range(10)
  batch_size = 3
  batched_data = cb.make_batch(data, batch_size)
  for batch in batched_data:
      print(batch)
  # >>> [0, 1, 2]
  # >>> [3, 4, 5]
  # >>> [6, 7, 8]
  # >>> [9]
  ```
