# make_batch

> [make_batch(data: Union[Iterable, Generator], batch_size: int) -> Generator[List, None, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/utils.py#L13)

- **說明**：這個函數用於將數據轉換為批次數據。

- **參數**

  - **data** (`Union[Iterable, Generator]`)：數據生成器。
  - **batch_size** (`int`)：批次數據的大小。

- **傳回值**

  - **Generator[List, None, None]**：批次數據的生成器。

- **範例**

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
