# imwarp_quadrangle

> [imwarp_quadrangle(img: np.ndarray, polygon: Union[Polygon, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L155)

- **説明**：入力画像に、与えられた多角形で定義された 4 点透視変換を適用します。この関数は自動的に 4 点の順番をソートします。順番は、最初の点が左上、2 番目が右上、3 番目が右下、4 番目が左下となります。画像変換のターゲットサイズは、多角形の最小回転外接矩形の幅と高さによって決まります。

- **パラメータ**

  - **img** (`np.ndarray`)：変換を行う入力画像。
  - **polygon** (`Union[Polygon, np.ndarray]`)：変換を定義する 4 つの点を含む多角形オブジェクト。

- **戻り値**

  - **np.ndarray**：変換後の画像。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('./resource/test_warp.jpg')
  polygon = cb.Polygon([[602, 404], [1832, 530], [1588, 985], [356, 860]])
  warp_img = cb.imwarp_quadrangle(img, polygon)
  ```

  ![imwarp_quadrangle](./resource/test_imwarp_quadrangle.jpg)

  上図では、緑色の枠が元の多角形範囲を示しています。
