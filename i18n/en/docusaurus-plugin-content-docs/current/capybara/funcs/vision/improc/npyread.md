# npyread

> [npyread(path: str | Path) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

- **Description**: Reads a numpy `.npy` file.

- **Parameters**

  - **path** (`str | Path`): Path to the `.npy` file.

- **Returns**

  - **np.ndarray | None**: Loaded array; returns `None` on failure.

- **Example**

  ```python
  from capybara.vision.improc import npyread

  img = npyread('lena.npy')
  ```
