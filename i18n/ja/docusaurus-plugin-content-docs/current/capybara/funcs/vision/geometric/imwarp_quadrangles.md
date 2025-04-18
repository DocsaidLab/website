# imwarp_quadrangles

> [imwarp_quadrangles(img: np.ndarray, polygons: Union[Polygons, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/geometric.py#L206)

- **説明**：入力画像に、与えられた「複数」の多角形で定義された 4 点透視変換を適用します。この関数は自動的に 4 点の順番をソートします。順番は、最初の点が左上、2 番目が右上、3 番目が右下、4 番目が左下となります。画像変換のターゲットサイズは、多角形の最小回転外接矩形の幅と高さによって決まります。

- **パラメータ**

  - **img** (`np.ndarray`)：変換を行う入力画像。
  - **polygons** (`Union[Polygons, np.ndarray]`)：変換を定義する「複数」の四つの点を含む多角形オブジェクト。

- **戻り値**

  - **List[np.ndarray]**：変換後の画像リスト。

- **使用例**

  ```python
  import capybara as cb

  img = cb.imread('./resource/test_warp.jpg')
  polygons = cb.Polygons([[[602, 404], [1832, 530], [1588, 985], [356, 860]]])
  warp_imgs = cb.imwarp_quadrangles(img, polygons)
  ```

  図示については [**imwarp_quadrangle**](./imwarp_quadrangle.md) を参照してください。
