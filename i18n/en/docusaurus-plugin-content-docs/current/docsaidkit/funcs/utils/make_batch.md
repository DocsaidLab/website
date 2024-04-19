---
sidebar_position: 19
---

# make_batch

>[make_batch(data: Union[Iterable, Generator], batch_size: int) -> Generator[List, None, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/utils.py#L13)

- **Description**: This function is used to convert data into batched data.

- **Parameters**:
    - **data** (`Union[Iterable, Generator]`): The data generator.
    - **batch_size** (`int`): The size of batches.

- **Returns**:
    - **Generator[List, None, None]**: A generator for batched data.

- **Example**:

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
