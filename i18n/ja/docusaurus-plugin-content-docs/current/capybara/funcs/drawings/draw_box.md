# draw_box

> [draw_box(img: np.ndarray, box: _Box, color: _Color = (0, 255, 0), thickness: _Thickness = 2) -> np.ndarray](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/visualization/draw.py)

- **依存関係**

  - 描画機能は optional extra です。先にインストールしてください：

    ```bash
    pip install "capybara-docsaid[visualization]"
    ```

- **説明**：画像にバウンディングボックスを描画します。

- **パラメータ**

  - **img** (`np.ndarray`)：描画する画像、NumPy 配列形式です。
  - **box** (`Union[Box, np.ndarray]`)：描画するバウンディングボックス、Box オブジェクトまたは NumPy 配列形式の[x1, y1, x2, y2]で指定します。
  - **color** (`_Color`)：描画する枠線の色。デフォルトは(0, 255, 0)です。
  - **thickness** (`_Thickness`)：枠線の太さ。デフォルトは 2 です。

- **戻り値**

  - **np.ndarray**：バウンディングボックスを描画した画像。

- **例**

  ```python
  from capybara import Box, imread
  from capybara.vision.visualization.draw import draw_box

  img = imread('lena.png')
  box = Box([20, 20, 100, 100])
  box_img = draw_box(img, box, color=(0, 255, 0), thickness=2)
  ```

  ![draw_box](./resource/test_draw_box.jpg)
