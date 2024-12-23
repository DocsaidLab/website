# draw_polygons

> [draw_polygons(img: np.ndarray, polygons: Polygons, color: \_Colors = (0, 255, 0), thickness: \_Thicknesses = 2, fillup=False, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L150)

- **説明**：画像に複数の多角形を描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式です。
  - **polygons** (`List[Union[Polygon, np.ndarray]]`)：描画する多角形のリスト、多角形オブジェクトまたは NumPy 配列形式の[[x1, y1], [x2, y2], ...]で指定します。
  - **color** (`_Colors`)：描画する多角形の色。単一の色または色のリストで指定できます。デフォルトは(0, 255, 0)です。
  - **thickness** (`_Thicknesses`)：描画する多角形の辺の太さ。単一の太さまたは太さのリストで指定できます。デフォルトは 2 です。
  - **fillup** (`bool`)：多角形を塗りつぶすかどうか。デフォルトは False です。
  - **kwargs**：その他のパラメータ。

- **戻り値**

  - **np.ndarray**：複数の多角形を描画した画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  polygons = [
      cb.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)]),
      cb.Polygon([(100, 100), (20, 100), (40, 40), (100, 80)])
  ]
  polygons_img = cb.draw_polygons(img, polygons, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
  ```

  ![draw_polygons](./resource/test_draw_polygons.jpg)
