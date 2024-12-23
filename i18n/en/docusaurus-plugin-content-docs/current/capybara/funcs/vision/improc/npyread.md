# npyread

> [npyread(path: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L174)

- **Description**: Reads an image array from a NumPy `.npy` file.

- **Parameters**:

  - **path** (`Union[str, Path]`): The path to the `.npy` file.

- **Return value**:

  - **np.ndarray**: The image array read from the file. Returns `None` if reading fails.

- **Example**:

  ```python
  import capybara as cb

  img = cb.npyread('lena.npy')
  ```
