# IpcamCapture

> [IpcamCapture(url: int, str, color_base: str) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/camera.py#L11)

- **説明**：IP カメラから画像をキャプチャします。

- **引数**

  - **url** (`int`, `str`)：ビデオソースの識別子。これはローカル接続のカメラデバイスインデックス、または IP カメラのネットワークアドレスを含む文字列です。ローカルカメラの場合、0 は通常デフォルトカメラです。デフォルトは 0。
  - **color_base** (`str`)：出力フレームの色空間。'BGR' または 'RGB' に設定できます。OpenCV の入力フレームは常に BGR 形式です。color_base が 'RGB' に設定されている場合、フレームは返される前に BGR から RGB に変換されます。デフォルトは 'BGR'。

- **属性**

  - **color_base** (`str`)：出力フレームの色空間。

- **メソッド**

  - **get_frame() -> np.ndarray**：現在キャプチャされたフレームを取得します。

- **例**

  ```python
  import capybara as cb

  cam = cb.IpcamCapture(url='http://your_ip:your_port/video')
  for frame in cam:
      cb.imwrite(frame, 'frame.jpg')
  ```
