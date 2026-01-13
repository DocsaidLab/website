# WebDemo

> [WebDemo(camera_ip: str, color_base: str = "BGR", route: str = "/", pipelines: list | None = None) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/ipcam/app.py)

- **依存関係**

  - `flask` が必要です（`capybara-docsaid[ipcam]` をインストールしてください）。

- **説明**：Web ページ上で IP カメラの映像を表示します。

- **引数**

  - **camera_ip** (`str`)：IP カメラの URL。
  - **color_base** (`str`)：出力フレームの色空間。'BGR' または 'RGB' に設定できます。OpenCV の入力フレームは常に BGR 形式です。color_base が 'RGB' に設定されている場合、フレームは返される前に BGR から RGB に変換されます。デフォルトは 'BGR'。
  - **route** (`str`)：Web ページのルート。デフォルトは '/'。
  - **pipelines** (`list | None`)：画像処理パイプライン。`None` の場合は `[]` として扱われます。

- **例**

  ```python
  from capybara.vision.ipcam.app import WebDemo

  WebDemo(camera_ip='http://your_ip:your_port/video').run()
  ```
