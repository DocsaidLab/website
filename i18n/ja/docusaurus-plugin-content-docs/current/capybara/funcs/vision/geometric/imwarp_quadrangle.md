---
sidebar_position: 4
---

# imwarp_quadrangle

> [imwarp_quadrangle(img: np.ndarray, polygon: Union[Polygon, np.ndarray]) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/geometric.py#L155C1-L203C71)

- **説明**：入力画像に対して、指定された多角形による 4 点透視変換を適用します。この関数は 4 点の順序を自動的にソートします。その順序は、最初の点が左上、2 番目が右上、3 番目が右下、4 番目が左下です。画像変換後のターゲットサイズは、多角形の最小回転外接矩形の長さと幅によって決まります。

- 引数

  - **img** (`np.ndarray`)：変換を行う入力画像。
  - **polygon** (`Union[Polygon, np.ndarray]`)：変換を定義する 4 つの点を含む多角形オブジェクト。

- **返り値**

  - **np.ndarray**：変換後の画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('./resource/test_warp.jpg')
  polygon = D.Polygon([[602, 404], [1832, 530], [1588, 985], [356, 860]])
  warp_img = D.imwarp_quadrangle(img, polygon)
  ```

  ![imwarp_quadrangle](./resource/test_imwarp_quadrangle.jpg)

  画像の緑色の枠が元の多角形の範囲を示しています。
