---
sidebar_position: 5
---

# draw_text

> [draw_text(img: np.ndarray, text: str, location: np.ndarray, color: tuple = (0, 0, 0), text_size: int = 12, font_path: str = None, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/visualization/draw.py#L211)

- **説明**：画像上の指定された位置に文字を描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像。
  - **text** (`str`)：描画する文字列。
  - **location** (`np.ndarray`)：文字列を描画する x, y 座標。
  - **color** (`tuple`)：文字の色の RGB 値。デフォルトは黒色(0, 0, 0)。
  - **text_size** (`int`)：描画する文字のサイズ。デフォルトは 12。
  - **font_path** (`str`)：使用するフォントファイルのパス。指定されていない場合、デフォルトのフォント「NotoSansMonoCJKtc-VF.ttf」が使用されます。
  - **kwargs**：描画時のその他のパラメータ（使用するライブラリやメソッドに依存）。

- **戻り値**

  - **np.ndarray**：文字が描画された画像。

- **例**

  ```python
  import docsaidkit as D

  img = D.imread('lena.png')
  text_img = D.draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
  ```

  ![draw_text](./resource/test_draw_text.jpg)
