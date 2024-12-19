---
sidebar_position: 3
---

# draw_polygon

> [draw_polygon(img: np.ndarray, polygon: Union[Polygon, np.ndarray], color: \_Color = (0, 255, 0), thickness: \_Thickness = 2, fillup=False, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L106)

- **説明**：画像上に多角形を描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式。
  - **polygon** (`Union[Polygon, np.ndarray]`)：描画する多角形、多角形オブジェクトまたは NumPy 配列形式の[[x1, y1], [x2, y2], ...]。
  - **color** (`_Color`)：描画する多角形の色。デフォルトは(0, 255, 0)。
  - **thickness** (`_Thickness`)：描画する多角形の辺の太さ。デフォルトは 2。
  - **fillup** (`bool`)：多角形を塗りつぶすかどうか。デフォルトは False。
  - **kwargs**：その他のパラメータ。

- **戻り値**

  - **np.ndarray**：多角形が描画された画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  polygon = D.Polygon([(20, 20), (100, 20), (80, 80), (20, 40)])
  polygon_img = D.draw_polygon(img, polygon, color=(0, 255, 0), thickness=2)
  ```

  ![draw_polygon](./resource/test_draw_polygon.jpg)
