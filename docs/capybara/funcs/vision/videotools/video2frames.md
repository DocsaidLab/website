# video2frames

> [video2frames(video_path: str | Path, frame_per_sec: int | None = None) -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/videotools/video2frames.py)

- **說明**：從視訊中提取幀（BGR）。支援的視訊格式有 `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, `.MKV`。

- **參數**

  - **video_path** (`str`)：視訊的路徑。
  - **frame_per_sec** (`int | None`)：每秒提取的幀數。如果為 `None`，則提取所有幀。

- **傳回值**

  - **list[np.ndarray]**：幀的列表。若視訊檔案無法打開，回傳空列表。

- **例外**

  - **TypeError**：`video_path` 不存在或副檔名不在支援清單內時。
  - **ValueError**：`frame_per_sec <= 0` 時。

- **範例**

  ```python
  import capybara as cb

  frames = cb.video2frames('video.mp4', frame_per_sec=1)
  for i, frame in enumerate(frames):
      cb.imwrite(frame, f'{i}.jpg')
  ```
