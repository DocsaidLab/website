# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: Union[Polygon, np.ndarray], color: \_Color = (0, 255, 0), thickness: \_Thickness = 2, fillup=False, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L103)

- **説明**：画像に多角形を描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式です。
  - **polygon** (`Union[Polygon, np.ndarray]`)：描画する多角形、多角形オブジェクトまたは NumPy 配列形式の[[x1, y1], [x2, y2], ...]で指定します。
  - **color** (`_Color`)：描画する多角形の色。デフォルトは(0, 255, 0)です。
  - **thickness** (`_Thickness`)：描画する多角形の辺の太さ。デフォルトは 2 です。
  - **fillup** (`bool`)：多角形を塗りつぶすかどうか。デフォルトは False です。
  - **kwargs**：その他のパラメータ。

- **戻り値**

  - **np.ndarray**：多角形を描画した画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  polygon = cb.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  polygon_img = cb.draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
  ```

  ![draw_polygon](./resource/test_draw_polygon.jpg)
