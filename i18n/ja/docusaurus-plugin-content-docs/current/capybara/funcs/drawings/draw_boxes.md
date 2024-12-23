# draw_boxes

> [draw_boxes(img: np.ndarray, boxes: Union[Boxes, np.ndarray], color: \_Colors = (0, 255, 0), thickness: \_Thicknesses = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L70)

- **説明**：画像に複数のバウンディングボックスを描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式です。
  - **boxes** (`Union[Boxes, np.ndarray]`)：描画するバウンディングボックス、Box オブジェクトのリストまたは NumPy 配列形式の[[x1, y1, x2, y2], ...]で指定します。
  - **color** (`_Colors`)：描画する枠線の色。単一の色または色のリストで指定できます。デフォルトは(0, 255, 0)です。
  - **thickness** (`_Thicknesses`)：枠線の太さ。単一の太さまたは太さのリストで指定できます。デフォルトは 2 です。

- **戻り値**

  - **np.ndarray**：バウンディングボックスを描画した画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  boxes = [cb.Box([20, 20, 100, 100]), cb.Box([150, 150, 200, 200])]
  boxes_img = cb.draw_boxes(img, boxes, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
  ```

  ![draw_boxes](./resource/test_draw_boxes.jpg)
