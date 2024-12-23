# img_to_md5

> [img_to_md5(img: np.ndarray) -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L42)

- **説明**：指定された画像に基づいて md5 を生成します。`gen_md5` と同じ理由で存在しますが、こちらは画像入力に特化した関数です。

- **引数**

  - **img** (`np.ndarray`)：画像データ。

- **戻り値**

  - **str**：md5 値。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  md5 = cb.img_to_md5(img)
  print(md5)
  # >>> 'd41d8cd98f00b204e9800998ecf8427e'
  ```
