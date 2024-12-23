# WebDemo

> [WebDemo(camera_ip: str, color_base: str = 'BGR', route: str = '/', pipelines: list = []) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/app.py#L23)

- **説明**：Web ページ上で IP カメラの映像を表示します。

- **引数**

  - **camera_ip** (`str`)：IP カメラの URL。
  - **color_base** (`str`)：出力フレームの色空間。'BGR' または 'RGB' に設定できます。OpenCV の入力フレームは常に BGR 形式です。color_base が 'RGB' に設定されている場合、フレームは返される前に BGR から RGB に変換されます。デフォルトは 'BGR'。
  - **route** (`str`)：Web ページのルート。デフォルトは '/'。
  - **pipelines** (`list`)：画像処理パイプライン。デフォルトは空のリスト。

- **例**

  ```python
  import capybara as cb

  cb.WebDemo(camera_ip='http://your_ip:your_port/video').run()
  ```
