# video2frames_v2

> [video2frames_v2(video_path: str | Any, frame_per_sec: int | None = None, start_sec: float = 0, end_sec: float | None = None, n_threads: int = 8, max_size: int = 1920, color_base: str = 'BGR') -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/videotools/video2frames_v2.py)

- **說明**：以多執行緒從視訊中抽取幀，並可選擇依秒取樣、裁切時間區間、縮放與色彩空間轉換。

- **參數**

  - **video_path** (`str | Any`)：視訊路徑（需存在且副檔名在支援清單內）。
  - **frame_per_sec** (`int | None`)：每秒取樣幀數。若為 `None`，會使用原始 FPS。
  - **start_sec** (`float`)：起始秒數。預設為 0。
  - **end_sec** (`float | None`)：結束秒數（不含）。若為 `None` 或超過影片長度，會截到影片結尾。
  - **n_threads** (`int`)：使用的執行緒數。預設為 8。
  - **max_size** (`int`)：輸出幀的長邊目標尺寸。預設為 1920。
  - **color_base** (`str`)：輸出色彩空間。預設為 `BGR`（例如可設 `RGB`）。

- **傳回值**

  - **list[np.ndarray]**：抽取出的影像幀列表（可能少於預期數量；遇到無法讀取的幀時會略過）。

- **例外**

  - **TypeError**：`video_path` 不存在或副檔名不在支援清單內。
  - **ValueError**：`n_threads < 1`、`frame_per_sec <= 0`、`start_sec > end_sec`，或要求的取樣數量超過該區間可用總幀數時。

- **備註**

  - `frame_per_sec` 大於原始 FPS 時，可能觸發 `ValueError`（取樣數量超過總幀數）。
  - 會將幀縮放到長邊約為 `max_size`（可能放大或縮小），再依 `color_base` 進行轉換。

- **範例**

  ```python
  import capybara as cb

  frames = cb.video2frames_v2(
      'video.mp4',
      frame_per_sec=2,
      start_sec=0,
      end_sec=10,
      n_threads=4,
      max_size=1280,
      color_base='BGR',
  )
  ```
