# draw_boxes

> [draw_boxes(img: np.ndarray, boxes: _Boxes, colors: _Colors = (0, 255, 0), thicknesses: _Thicknesses = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依存関係**

  - `capybara-docsaid[visualization]` を先にインストールしてください。

- **説明**：画像に複数のバウンディングボックスを描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式です。
  - **boxes** (`Union[Boxes, np.ndarray]`)：描画するバウンディングボックス、Box オブジェクトのリストまたは NumPy 配列形式の[[x1, y1, x2, y2], ...]で指定します。
  - **colors** (`_Colors`)：描画する枠線の色（BGR）。単一の色または色のリストで指定できます。デフォルトは(0, 255, 0)です。
  - **thicknesses** (`_Thicknesses`)：枠線の太さ。単一の太さまたは太さのリストで指定できます。デフォルトは 2 です。

- **戻り値**

  - **np.ndarray**：バウンディングボックスを描画した画像。

- **例**

  ```python
  from capybara import Box, imread
  from capybara.vision.visualization.draw import draw_boxes

  img = imread('lena.png')
  boxes = [Box([20, 20, 100, 100]), Box([150, 150, 200, 200])]
  boxes_img = draw_boxes(
      img,
      boxes,
      colors=[(0, 255, 0), (255, 0, 0)],
      thicknesses=2,
  )
  ```

  ![draw_boxes](./resource/test_draw_boxes.jpg)
