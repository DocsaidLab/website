# pad

> [pad(img: np.ndarray, pad_size: Union[int, Tuple[int, int], Tuple[int, int, int, int]], fill_value: Optional[Union[int, Tuple[int, int, int]]] = 0, pad_mode: Union[str, int, BORDER] = BORDER.CONSTANT) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L194)

- **說明**：對輸入影像進行填充處理。

- **參數**

  - **img** (`np.ndarray`)：要進行填充處理的輸入影像。
  - **pad_size** (`Union[int, Tuple[int, int], Tuple[int, int, int, int]]`)：填充大小。可以是整數以指定所有邊的相同填充量，也可以是元組`(pad_top, pad_bottom, pad_left, pad_right)`以指定每個邊的不同填充量，或者是元組`(pad_height, pad_width)`以指定高度和寬度的相同填充量。
  - **fill_value** (`Optional[Union[int, Tuple[int, int, int]]]`)：用於填充的值。如果輸入影像是彩色影像（3 通道），則 fill_value 可以是整數或元組`(R, G, B)`以指定填充的顏色。如果輸入影像是灰度影像（1 通道），則 fill_value 應為整數。預設為 0。
  - **pad_mode** (`Union[str, int, BORDER]`)：填充模式。可用選項有： - BORDER.CONSTANT：使用常數值（fill_value）進行填充。 - BORDER.REPLICATE：通過複製邊緣像素進行填充。 - BORDER.REFLECT：通過圍繞邊緣反射影像進行填充。 - BORDER.REFLECT101：通過圍繞邊緣反射影像進行填充，並進行輕微調整以避免產生人工痕跡。
    預設為 BORDER.CONSTANT。

- **傳回值**

  - **np.ndarray**：填充後的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  pad_img = cb.pad(img, pad_size=20, fill_value=(255, 0, 0))

  # Resize the padded image to the original size for visualization
  pad_img = cb.imresize(pad_img, [img.shape[0], img.shape[1]])
  ```

  ![pad](./resource/test_pad.jpg)
