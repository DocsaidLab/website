# IpcamCapture

> [IpcamCapture(url: int, str, color_base: str) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/camera.py#L11)

- **說明**：從 IP 攝影機中捕獲影像。

- **參數**

  - **url** (`int`, `str`)：視訊源的辨識符。它可以是本地連接攝影機的設備索引，也可以是包含 IP 攝影機網路地址的字串。對於本地攝影機，0 通常是默認攝影機。默認為 0。
  - **color_base** (`str`)：輸出幀的顏色空間。它可以是 'BGR' 或 'RGB'。請注意，OpenCV 的輸入幀始終是 BGR 格式。如果 color_base 設置為 'RGB'，則每個幀將在返回之前從 BGR 轉換為 RGB。默認為 'BGR'。

- **屬性**

  - **color_base** (`str`)：輸出幀的顏色空間。

- **方法**

  - **get_frame() -> np.ndarray**：獲取當前捕獲的幀。

- **範例**

  ```python
  import capybara as cb

  cam = cb.IpcamCapture(url='http://your_ip:your_port/video')
  for frame in cam:
      cb.imwrite(frame, 'frame.jpg')
  ```
