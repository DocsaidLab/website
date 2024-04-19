---
sidebar_position: 6
---

# Tqdm

>[Tqdm(iterable=None, desc=None, smoothing=0, **kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/custom_tqdm.py#L8)

- **Description**: This is a custom progress bar inherited from `tqdm`, used to display progress during iteration over an iterable. The modification we made to the original `tqdm` is in the `total` parameter. When the user does not specify `total`, we automatically calculate the length of the `iterable` and set it as `total`. This design allows users to correctly display the progress bar without needing to specify `total`.

- **Parameters**:
    - **iterable** (`Iterable`): The object to iterate over.
    - **desc** (`str`): Description of the progress bar.
    - **smoothing** (`int`): Smoothing parameter.
    - **kwargs** (`Any`): Other parameters.

- **Example**:

    ```python
    import docsaidkit as D

    for i in D.Tqdm(range(100), desc='Processing'):
        pass
    ```
