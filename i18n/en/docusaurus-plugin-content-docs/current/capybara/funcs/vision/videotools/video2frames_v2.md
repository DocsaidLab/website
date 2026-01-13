# video2frames_v2

> [video2frames_v2(video_path: str | Any, frame_per_sec: int | None = None, start_sec: float = 0, end_sec: float | None = None, n_threads: int = 8, max_size: int = 1920, color_base: str = 'BGR') -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/videotools/video2frames_v2.py)

- **Description**: Extracts frames from a video using multi-threading, with optional per-second sampling, time range cropping, resizing, and color conversion.

- **Parameters**

  - **video_path** (`str | Any`): Video path (must exist and have a supported suffix).
  - **frame_per_sec** (`int | None`): Sample rate (frames per second). If `None`, uses the original FPS.
  - **start_sec** (`float`): Start time in seconds. Default is 0.
  - **end_sec** (`float | None`): End time in seconds (exclusive). If `None` or beyond the video length, it will be clamped to the end.
  - **n_threads** (`int`): Number of threads. Default is 8.
  - **max_size** (`int`): Target long-edge size of output frames. Default is 1920.
  - **color_base** (`str`): Output color space. Default is `BGR` (e.g. set to `RGB`).

- **Returns**

  - **list[np.ndarray]**: Extracted frames (may be fewer than expected; unreadable frames are skipped).

- **Exceptions**

  - **TypeError**: `video_path` does not exist or the suffix is not supported.
  - **ValueError**: `n_threads < 1`, `frame_per_sec <= 0`, `start_sec > end_sec`, or the requested sample count exceeds the total frames in the range.

- **Notes**

  - If `frame_per_sec` is larger than the original FPS, it may raise `ValueError` (sample count exceeds total frames).
  - Frames are resized to make the long edge approximately `max_size` (may upscale or downscale), then converted to `color_base`.

- **Example**

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
