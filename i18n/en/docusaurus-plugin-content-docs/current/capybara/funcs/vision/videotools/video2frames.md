# video2frames

> [video2frames(video_path: str, frame_per_sec: int = None) -> List[np.ndarray]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/videotools/video2frames.py#L19)

- **Description**: Extracts frames from a video. Supported video formats include `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, and `.MKV`.

- **Parameters**

  - **video_path** (`str`): The path to the video file.
  - **frame_per_sec** (`int`): The number of frames to extract per second. If `None`, all frames are extracted.

- **Returns**

  - **List[np.ndarray]**: A list of frames.

- **Example**

  ```python
  import capybara as cb

  frames = cb.video2frames('video.mp4', frame_per_sec=1)
  for i, frame in enumerate(frames):
      cb.imwrite(frame, f'{i}.jpg')
  ```
