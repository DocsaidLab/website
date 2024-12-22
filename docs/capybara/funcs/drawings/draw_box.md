# draw_box

> [draw_box(img: np.ndarray, box: Union[Box, np.ndarray], color: \_Color = (0, 255, 0), thickness: \_Thickness = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L36)

- **說明**：在影像上繪製 Bounding Box。

- **參數**

  - **img** (`np.ndarray`)：要繪製的影像，為 NumPy 陣列。
  - **box** (`Union[Box, np.ndarray]`)：要繪製的 Bounding Box，可以是 Box 物件或 NumPy 陣列形式的 [x1, y1, x2, y2]。
  - **color** (`_Color`)：要繪製的框的顏色。預設為 (0, 255, 0)。
  - **thickness** (`_Thickness`)：要繪製的框線的粗細。預設為 2。

- **傳回值**

  - **np.ndarray**：繪製了框的影像。

- **範例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  box = cb.Box([20, 20, 100, 100])
  box_img = cb.draw_box(img, box, color=(0, 255, 0), thickness=2)
  ```

  ![draw_box](./resource/test_draw_box.jpg)
