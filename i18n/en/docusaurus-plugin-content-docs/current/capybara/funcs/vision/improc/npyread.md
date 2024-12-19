---
sidebar_position: 9
---

# npyread

> [npyread(path: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/improc.py#L174)

- **Description**: Read an image array from a NumPy `.npy` file.

- **Parameters**
    - **path** (`Union[str, Path]`): The path to the `.npy` file.

- **Returns**
    - **np.ndarray**: The read image array. Returns `None` if reading fails.

- **Example**

    ```python
    import docsaidkit as D

    img = D.npyread('lena.npy')
    ```
