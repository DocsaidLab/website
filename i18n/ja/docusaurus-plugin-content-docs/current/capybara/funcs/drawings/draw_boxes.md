---
sidebar_position: 2
---

# draw_boxes

> [draw_boxes(img: np.ndarray, boxes: Union[Boxes, np.ndarray], color: \_Colors = (0, 255, 0), thickness: \_Thicknesses = 2) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L67)

- **説明**：画像上に複数の Bounding Box を描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式。
  - **boxes** (`Union[Boxes, np.ndarray]`)：描画する Bounding Box、Box オブジェクトのリストまたは NumPy 配列形式の[[x1, y1, x2, y2], ...]。
  - **color** (`_Colors`)：描画する枠線の色。単一の色または色のリスト。デフォルトは(0, 255, 0)。
  - **thickness** (`_Thicknesses`)：描画する枠線の太さ。単一の太さまたは太さのリスト。デフォルトは 2。

- **戻り値**

  - **np.ndarray**：枠線が描画された画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  boxes = [D.Box([20, 20, 100, 100]), D.Box([150, 150, 200, 200])]
  boxes_img = D.draw_boxes(img, boxes, color=[(0, 255, 0), (255, 0, 0)], thickness=2)
  ```

  ![draw_boxes](./resource/test_draw_boxes.jpg)
