# video2frames_v2

> [video2frames_v2(video_path: str | Any, frame_per_sec: int | None = None, start_sec: float = 0, end_sec: float | None = None, n_threads: int = 8, max_size: int = 1920, color_base: str = 'BGR') -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/videotools/video2frames_v2.py)

- **説明**：マルチスレッドで動画からフレームを抽出します。秒間サンプリング、時間範囲の切り出し、リサイズ、色空間変換に対応します。

- **パラメータ**

  - **video_path** (`str | Any`)：動画パス（存在し、対応 suffix を持つ必要があります）。
  - **frame_per_sec** (`int | None`)：サンプルレート（fps）。`None` の場合は元の FPS を使用します。
  - **start_sec** (`float`)：開始秒。デフォルトは 0。
  - **end_sec** (`float | None`)：終了秒（exclusive）。`None` または動画長を超える場合、末尾に丸められます。
  - **n_threads** (`int`)：スレッド数。デフォルトは 8。
  - **max_size** (`int`)：出力フレームの長辺サイズ。デフォルトは 1920。
  - **color_base** (`str`)：出力色空間。デフォルトは `BGR`（例：`RGB` を指定）。

- **戻り値**

  - **list[np.ndarray]**：抽出されたフレーム（読み取り不能なフレームはスキップされるため、期待数より少ない場合があります）。

- **例外**

  - **TypeError**：`video_path` が存在しない、または suffix が未対応。
  - **ValueError**：`n_threads < 1`、`frame_per_sec <= 0`、`start_sec > end_sec`、または要求サンプル数が範囲内の総フレーム数を超える場合。

- **備考**

  - `frame_per_sec` が元の FPS を上回る場合、`ValueError` になることがあります（サンプル数が総フレーム数を超える）。
  - フレームは長辺が概ね `max_size` になるようにリサイズ（拡大/縮小あり）した後、`color_base` に変換されます。

- **例**

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

