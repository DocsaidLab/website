# npyread

> [npyread(path: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L174)

- **說明**：從 NumPy `.npy` 文件中讀取圖像數組。

- **參數**

  - **path** (`Union[str, Path]`)：`.npy` 文件的路徑。

- **傳回值**

  - **np.ndarray**：讀取的圖像數組。如果讀取失敗，則返回 `None`。

- **範例**

  ```python
  import capybara as cb

  img = cb.npyread('lena.npy')
  ```
