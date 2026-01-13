# video2frames

> [video2frames(video_path: str | Path, frame_per_sec: int | None = None) -> list[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/videotools/video2frames.py)

- **Description**: Extracts frames from a video. Supported video formats include `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, and `.MKV`.

- **Parameters**

  - **video_path** (`str | Path`): The path to the video file.
  - **frame_per_sec** (`int | None`): The number of frames to extract per second. If `None`, all frames are extracted.

- **Returns**

  - **list[np.ndarray]**: A list of frames. Returns an empty list if the video cannot be opened.

- **Exceptions**

  - **TypeError**: `video_path` does not exist or has an unsupported file extension.
  - **ValueError**: `frame_per_sec <= 0`.

- **Example**

  ```python
  import capybara as cb

  frames = cb.video2frames('video.mp4', frame_per_sec=1)
  for i, frame in enumerate(frames):
      cb.imwrite(frame, f'{i}.jpg')
  ```
