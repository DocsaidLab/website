# draw_text

> [draw_text(img: np.ndarray, text: str, location: _Point, color: _Color = (0, 0, 0), text_size: int = 12, font_path: str | Path | None = None, **kwargs) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依賴**

  - 請先安裝 `capybara-docsaid[visualization]`。

- **說明**：在影像上指定位置繪製文字。

- **參數**

  - **img** (`np.ndarray`)：要繪製的影像。
  - **text** (`str`)：要繪製的文字。
  - **location** (`np.ndarray`)：文字應繪製的 x, y 座標。
  - **color** (`tuple`)：文字顏色（BGR）。預設為 (0, 0, 0)。
  - **text_size** (`int`)：要繪製的文字大小。預設為 12。
  - **font_path** (`str`)：要使用的字型檔案的路徑。如果未提供，則使用預設字型 "NotoSansMonoCJKtc-VF.ttf"。
  - **kwargs**：繪製時的其他參數，取決於所使用的底層函式庫或方法。

- **傳回值**

  - **np.ndarray**：繪製了文字的影像。

- **範例**

  ```python
  from capybara import imread
  from capybara.vision.visualization.draw import draw_text

  img = imread('lena.png')
  text_img = draw_text(img, 'Hello, Docsaid!', location=(20, 20), color=(0, 255, 0), text_size=12)
  ```

  ![draw_text](./resource/test_draw_text.jpg)
