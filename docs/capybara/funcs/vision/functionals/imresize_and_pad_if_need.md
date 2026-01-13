# imresize_and_pad_if_need

> [imresize_and_pad_if_need(img: np.ndarray, max_h: int, max_w: int, interpolation: str | int | INTER = INTER.BILINEAR, pad_value: int | tuple[int, int, int] | None = 0, pad_mode: str | int | BORDER = BORDER.CONSTANT, return_scale: bool = False) -> np.ndarray | tuple[np.ndarray, float]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **說明**：將影像縮放到不超過 `(max_h, max_w)`，並在需要時補齊到固定大小。

- **參數**

  - **img** (`np.ndarray`)：輸入影像。
  - **max_h** (`int`)：輸出影像高度上限（同時也是補齊後的固定高度）。
  - **max_w** (`int`)：輸出影像寬度上限（同時也是補齊後的固定寬度）。
  - **interpolation** (`str | int | INTER`)：縮放插值方法。預設為 `INTER.BILINEAR`。
  - **pad_value** (`int | tuple[int, int, int] | None`)：補齊像素值。對 3-channel 影像可用單一整數或 tuple（OpenCV 慣例：BGR）。預設為 0。
  - **pad_mode** (`str | int | BORDER`)：補齊模式。預設為 `BORDER.CONSTANT`。
  - **return_scale** (`bool`)：是否回傳縮放比例。預設為 `False`。

- **傳回值**

  - `return_scale=False`：回傳 `np.ndarray`。
  - `return_scale=True`：回傳 `(np.ndarray, float)`，其中 float 為 `scale = min(max_h/raw_h, max_w/raw_w)`。

- **備註**

  - 只會在下方與右方進行補齊（top=0, left=0）。
  - `max_h/max_w` 小於原圖時會縮小；大於原圖時會放大。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')

  out, scale = cb.imresize_and_pad_if_need(
      img,
      max_h=640,
      max_w=640,
      pad_value=0,
      return_scale=True,
  )
  print(out.shape, scale)
  ```
