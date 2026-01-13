# pad

> [pad(img: np.ndarray, pad_size: int | tuple[int, int] | tuple[int, int, int, int], pad_value: int | tuple[int, ...] | None = 0, pad_mode: str | int | BORDER = BORDER.CONSTANT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/functionals.py)

- **說明**：對輸入影像進行填充處理。

- **參數**

  - **img** (`np.ndarray`)：要進行填充處理的輸入影像。
  - **pad_size** (`Union[int, Tuple[int, int], Tuple[int, int, int, int]]`)：填充大小。可以是整數以指定所有邊的相同填充量，也可以是元組`(pad_top, pad_bottom, pad_left, pad_right)`以指定每個邊的不同填充量，或者是元組`(pad_height, pad_width)`以指定高度和寬度的相同填充量。
  - **pad_value** (`int | tuple[int, ...] | None`)：用於填充的值。對 3-channel 影像可用單一整數或 tuple（OpenCV 慣例：BGR）；對灰階影像須為整數。預設為 0。
  - **pad_mode** (`str | int | BORDER`)：填充模式。可用選項有： - BORDER.CONSTANT：使用常數值（pad_value）進行填充。 - BORDER.REPLICATE：通過複製邊緣像素進行填充。 - BORDER.REFLECT：通過圍繞邊緣反射影像進行填充。 - BORDER.REFLECT_101：通過圍繞邊緣反射影像進行填充，並進行輕微調整以避免產生人工痕跡。
    預設為 BORDER.CONSTANT。

- **傳回值**

  - **np.ndarray**：填充後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  pad_img = cb.pad(img, pad_size=20, pad_value=(255, 0, 0))

  # Resize the padded image to the original size for visualization
  pad_img = cb.imresize(pad_img, [img.shape[0], img.shape[1]])
  ```

  ![pad](./resource/test_pad.jpg)
