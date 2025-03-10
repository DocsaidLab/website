# WebDemo

> [WebDemo(camera_ip: str, color_base: str = 'BGR', route: str = '/', pipelines: list = []) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/app.py#L23)

- **說明**：在網頁上展示 IP 攝影機的影像。

- **參數**

  - **camera_ip** (`str`)：IP 攝影機的網址。
  - **color_base** (`str`)：輸出幀的顏色空間。它可以是 'BGR' 或 'RGB'。請注意，OpenCV 的輸入幀始終是 BGR 格式。如果 color_base 設置為 'RGB'，則每個幀將在返回之前從 BGR 轉換為 RGB。默認為 'BGR'。
  - **route** (`str`)：網頁路由。默認為 '/'。
  - **pipelines** (`list`)：影像處理管道。默認為空列表。

- **範例**

  ```python
  import capybara as cb

  cb.WebDemo(camera_ip='http://your_ip:your_port/video').run()
  ```
