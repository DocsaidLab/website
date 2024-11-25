---
sidebar_position: 1
---

# video2frames

> [video2frames(video_path: str, frame_per_sec: int = None) -> List[np.ndarray]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/videotools/video2frames.py#L19)

- **説明**：動画からフレームを抽出します。サポートされている動画形式には `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, `.MKV` があります。

- 引数

  - **video_path** (`str`)：動画のファイルパス。
  - **frame_per_sec** (`int`)：1 秒あたりに抽出するフレーム数。`None`の場合、すべてのフレームを抽出します。

- **返り値**

  - **List[np.ndarray]**：フレームのリスト。

- **例**

  ```python
  import docsaidkit as D

  frames = D.video2frames('video.mp4', frame_per_sec=1)
  for frame in frames:
      D.imshow(frame)
  ```
