# IpcamCapture

> [IpcamCapture(url: int | str = 0, color_base: str = "BGR") -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/ipcam/camera.py)

- **依存関係**

  - `ipcam` は optional extra です。`WebDemo` を使う場合は先にインストールしてください：

    ```bash
    pip install "capybara-docsaid[ipcam]"
    ```

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
  from capybara import imwrite
  from capybara.vision.ipcam.camera import IpcamCapture

  cam = IpcamCapture(url='http://your_ip:your_port/video')
  for frame in cam:
      imwrite(frame, 'frame.jpg')
  ```
