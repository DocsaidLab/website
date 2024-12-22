# video2frames

> [video2frames(video_path: str, frame_per_sec: int = None) -> List[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/videotools/video2frames.py#L19)

- **說明**：從視訊中提取幀。其中可以支援的視訊格式有 `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, `.MKV`。

- **參數**

  - **video_path** (`str`)：視訊的路徑。
  - **frame_per_sec** (`int`)：每秒提取的幀數。如果為 `None`，則提取所有幀。

- **傳回值**

  - **List[np.ndarray]**：幀的列表。

- **範例**

  ```python
  import capybara as cb

  frames = cb.video2frames('video.mp4', frame_per_sec=1)
  for i, frame in enumerate(frames):
      cb.imwrite(frame, f'{i}.jpg')
  ```
