# imcropbox

> [imcropbox(img: np.ndarray, box: Union[Box, np.ndarray], use_pad: bool = False) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L257)

- **說明**：使用提供的框裁剪輸入影像。

- **參數**

  - **img** (`np.ndarray`)：要裁剪的輸入影像。
  - **box** (`Union[Box, np.ndarray]`)：裁剪框。輸入可以為 Capybara 自定義的 Box 物件，由 (x1, y1, x2, y2) 座標定義，也可以是具有相同格式的 NumPy 陣列。
  - **use_pad** (`bool`)：是否使用填充來處理超出邊界的區域。如果設置為 True，則外部區域將使用零填充。預設為 False。

- **傳回值**

  - **np.ndarray**：裁剪後的影像。

- **範例**

  ```python
  import capybara as cb

  # 使用自定義 Box 物件
  img = cb.imread('lena.png')
  box = cb.Box([50, 50, 200, 200], box_mode='xyxy')
  cropped_img = cb.imcropbox(img, box, use_pad=True)

  # Resize the cropped image to the original size for visualization
  cropped_img = cb.imresize(cropped_img, [img.shape[0], img.shape[1]])
  ```

  ![imcropbox_box](./resource/test_imcropbox.jpg)
