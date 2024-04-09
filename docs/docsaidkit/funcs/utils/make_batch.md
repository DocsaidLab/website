---
sidebar_position: 19
---

# make_batch

>[make_batch(data: Union[Iterable, Generator], batch_size: int) -> Generator[List, None, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L13)

- **說明**：這個函數用於將數據轉換為批次數據。

- **參數**
    - **data** (`Union[Iterable, Generator]`)：數據生成器。
    - **batch_size** (`int`)：批次數據的大小。

- **傳回值**
    - **Generator[List, None, None]**：批次數據的生成器。

- **範例**

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

