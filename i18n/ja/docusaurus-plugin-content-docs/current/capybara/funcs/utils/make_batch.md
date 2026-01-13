# make_batch

> [make_batch(data: Iterable | Generator, batch_size: int) -> Generator[list, None, None]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/utils.py)

- **説明**：この関数はデータをバッチデータに変換するために使用されます。

- **引数**

  - **data** (`Union[Iterable, Generator]`)：データ生成器。
  - **batch_size** (`int`)：バッチデータのサイズ。

- **戻り値**

  - **Generator[List, None, None]**：バッチデータの生成器。

- **例**

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
