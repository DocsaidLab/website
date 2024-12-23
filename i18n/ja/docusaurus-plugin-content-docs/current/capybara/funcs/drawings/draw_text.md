# draw_text

> [draw_text(img: np.ndarray, text: str, location: np.ndarray, color: tuple = (0, 0, 0), text_size: int = 12, font_path: str = None, \*\*kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/visualization/draw.py#L197)

- **説明**：画像上の指定された位置にテキストを描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像。
  - **text** (`str`)：描画するテキスト。
  - **location** (`np.ndarray`)：テキストを描画する x, y 座標。
  - **color** (`tuple`)：テキストの色の RGB 値。デフォルトは黒(0, 0, 0)です。
  - **text_size** (`int`)：描画するテキストのサイズ。デフォルトは 12 です。
  - **font_path** (`str`)：使用するフォントファイルのパス。指定しない場合、デフォルトのフォント "NotoSansMonoCJKtc-VF.ttf" が使用されます。
  - **kwargs**：描画時のその他のパラメータ（使用する基盤ライブラリやメソッドに依存します）。

- **戻り値**

  - **np.ndarray**：テキストを描画した画像。

- **例**

  ```python
  import capybara as cb

  img = cb.imread('lena.png')
  text_img = cb.draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
  ```

  ![draw_text](./resource/test_draw_text.jpg)
