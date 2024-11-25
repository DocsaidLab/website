---
sidebar_position: 19
---

# make_batch

> [make_batch(data: Union[Iterable, Generator], batch_size: int) -> Generator[List, None, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L13)

- **説明**：この関数はデータをバッチデータに変換するために使用されます。

- **パラメータ**

  - **data** (`Union[Iterable, Generator]`)：データジェネレーター。
  - **batch_size** (`int`)：バッチデータのサイズ。

- **戻り値**

  - **Generator[List, None, None]**：バッチデータのジェネレーター。

- **例**

  ```python
  import docsaidkit as D

  data = range(10)
  batch_size = 3
  batched_data = D.make_batch(data, batch_size)
  for batch in batched_data:
      print(batch)
  # >>> [0, 1, 2]
  # >>> [3, 4, 5]
  # >>> [6, 7, 8]
  # >>> [9]
  ```
