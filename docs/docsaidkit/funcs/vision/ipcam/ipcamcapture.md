---
sidebar_position: 1
---

# IpcamCapture

>[IpcamCapture(url: int, str, color_base: str) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/ipcam/camera.py#L11)

- **說明**：從 IP 攝影機中捕獲影像。

- **參數**
    - **url** (`int`, `str`)：視訊源的識別符。它可以是本地連接攝影機的設備索引，也可以是包含 IP 攝影機網絡地址的字符串。對於本地攝影機，0通常是默認攝影機。默認為 0。
    - **color_base** (`str`)：輸出幀的顏色空間。它可以是 'BGR' 或 'RGB'。請注意，OpenCV 的輸入幀始終是 BGR 格式。如果 color_base 設置為 'RGB'，則每個幀將在返回之前從 BGR 轉換為 RGB。默認為 'BGR'。

- **屬性**
    - **color_base** (`str`)：輸出幀的顏色空間。

- **方法**
    - **get_frame() -> np.ndarray**：獲取當前捕獲的幀。

- **範例**

    ```python
    import docsaidkit as D

    cam = D.IpcamCapture(url='http://your_ip:your_port/video')
    for frame in cam:
        D.imshow(frame)
    ```
