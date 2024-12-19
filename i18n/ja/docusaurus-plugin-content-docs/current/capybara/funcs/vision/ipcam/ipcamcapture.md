---
sidebar_position: 1
---

# IpcamCapture

> [IpcamCapture(url: int, str, color_base: str) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/ipcam/camera.py#L11)

- **説明**：IP カメラから画像をキャプチャします。

- 引数

  - **url** (`int`, `str`)：ビデオソースの識別子。ローカル接続カメラのデバイスインデックス、または IP カメラのネットワークアドレスを含む文字列で指定できます。ローカルカメラの場合、通常は 0 がデフォルトカメラです。デフォルトは 0 です。
  - **color_base** (`str`)：出力フレームの色空間。'BGR'または'RGB'を指定できます。OpenCV の入力フレームは常に BGR 形式です。`color_base`が'RGB'に設定されている場合、各フレームは返す前に BGR から RGB に変換されます。デフォルトは'BGR'です。

- 属性

  - **color_base** (`str`)：出力フレームの色空間。

- メソッド

  - **get_frame() -> np.ndarray**：現在キャプチャされたフレームを取得します。

- 例

  ```python
  import docsaidkit as D

  cam = D.IpcamCapture(url='http://your_ip:your_port/video')
  for frame in cam:
      D.imshow(frame)
  ```
