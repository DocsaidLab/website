# draw_text

> [draw_text(img: np.ndarray, text: str, location: _Point, color: _Color = (0, 0, 0), text_size: int = 12, font_path: str | Path | None = None, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依存関係**

  - `capybara-docsaid[visualization]` を先にインストールしてください。

- **説明**：画像上の指定された位置にテキストを描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像。
  - **text** (`str`)：描画するテキスト。
  - **location** (`_Point`)：テキストを描画する (x, y) 座標。
  - **color** (`_Color`)：テキストの色（BGR）。デフォルトは黒(0, 0, 0)です。
  - **text_size** (`int`)：描画するテキストのサイズ。デフォルトは 12 です。
  - **font_path** (`str | Path | None`)：使用するフォントファイルのパス。未指定の場合、組み込みフォント（`NotoSansMonoCJKtc-VF.ttf`）→ `PIL.ImageFont.load_default()` の順にフォールバックします。
  - **kwargs**：描画時のその他のパラメータ（使用する基盤ライブラリやメソッドに依存します）。

- **戻り値**

  - **np.ndarray**：テキストを描画した画像。

- **例**

  ```python
  from capybara import imread
  from capybara.vision.visualization.draw import draw_text

  img = imread('lena.png')
  text_img = draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
  ```

  ![draw_text](./resource/test_draw_text.jpg)
