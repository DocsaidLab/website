---
sidebar_position: 2
---

# WebDemo

> [WebDemo(camera_ip: str, color_base: str = 'BGR', route: str = '/', pipelines: list = []) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/ipcam/app.py#L23)

- **説明**：IP カメラの画像をウェブページ上に表示します。

- 引数

  - **camera_ip** (`str`)：IP カメラの URL。
  - **color_base** (`str`)：出力フレームの色空間。'BGR'または'RGB'を指定できます。OpenCV の入力フレームは常に BGR 形式です。`color_base`が'RGB'に設定されている場合、各フレームは返す前に BGR から RGB に変換されます。デフォルトは'BGR'です。
  - **route** (`str`)：ウェブページのルート。デフォルトは'/'。
  - **pipelines** (`list`)：画像処理パイプライン。デフォルトは空のリストです。

- 例

  ```python
  import docsaidkit as D

  D.WebDemo(camera_ip='http://your_ip:your_port/video').run()
  ```
