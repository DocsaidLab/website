# imcropboxes

> [imcropboxes(img: np.ndarray, boxes: Union[Box, np.ndarray], use_pad: bool = False) -> List[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L325)

- **說明**：使用提供多個框裁剪輸入影像。

- **參數**

  - **img** (`np.ndarray`)：要裁剪的輸入影像。
  - **boxes** (`Union[Boxes, np.ndarray]`)：裁剪框。輸入可以為 Capybara 自定義的 Boxes 物件，其定義為 List[Box]，也可以是具有相同格式的 NumPy 陣列。
  - **use_pad** (`bool`)：是否使用填充來處理超出邊界的區域。如果設置為 True，則外部區域將使用零填充。預設為 False。

- **傳回值**

  - **List[np.ndarray]**：裁剪後的影像列表。

- **範例**

  ```python
  import capybara as cb

  # 使用自定義 Box 物件
  img = cb.imread('lena.png')
  box1 = cb.Box([50, 50, 200, 200], box_mode='xyxy')
  box2 = cb.Box([50, 50, 100, 100], box_mode='xyxy')
  boxes = cb.Boxes([box1, box2])
  cropped_imgs = cb.imcropboxes(img, boxes, use_pad=True)

  # Resize the cropped image to the original size for visualization
  cropped_img = cb.imresize(cropped_img, [img.shape[0], img.shape[1]])
  ```

  ![imcropbox_boxes](./resource/test_imcropboxes.jpg)
