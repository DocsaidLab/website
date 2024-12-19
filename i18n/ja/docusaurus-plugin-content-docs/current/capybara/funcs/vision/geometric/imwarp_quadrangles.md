---
sidebar_position: 5
---

# imwarp_quadrangles

> [imwarp_quadrangles(img: np.ndarray, polygons: Union[Polygons, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L206)

- **説明**：入力画像に対して、指定された「複数」の多角形による 4 点透視変換を適用します。この関数は 4 点の順序を自動的にソートします。その順序は、最初の点が左上、2 番目が右上、3 番目が右下、4 番目が左下です。画像変換後のターゲットサイズは、多角形の最小回転外接矩形の長さと幅によって決まります。

- 引数

  - **img** (`np.ndarray`)：変換を行う入力画像。
  - **polygons** (`Union[Polygons, np.ndarray]`)：変換を定義する「複数」の 4 つの点を含む多角形オブジェクト。

- **返り値**

  - **List[np.ndarray]**：変換後の画像リスト。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('./resource/test_warp.jpg')
  polygons = D.Polygons([[[602, 404], [1832, 530], [1588, 985], [356, 860]]])
  warp_imgs = D.imwarp_quadrangles(img, polygons)
  ```

  詳細については[**imwarp_quadrangle**](./imwarp_quadrangle.md)を参照してください。
