# video2frames

> [video2frames(video_path: str, frame_per_sec: int = None) -> List[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/videotools/video2frames.py#L19)

- **説明**：動画からフレームを抽出します。サポートされている動画形式は `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, `.MKV` です。

- **引数**

  - **video_path** (`str`)：動画のパス。
  - **frame_per_sec** (`int`)：1 秒あたりに抽出するフレーム数。`None`の場合は、すべてのフレームを抽出します。

- **戻り値**

  - **List[np.ndarray]**：フレームのリスト。

- **例**

  ```python
  import capybara as cb

  frames = cb.video2frames('video.mp4', frame_per_sec=1)
  for i, frame in enumerate(frames):
      cb.imwrite(frame, f'{i}.jpg')
  ```
