---
sidebar_position: 1
---

# video2frames

```python
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

__all__ = ['video2frames', 'is_video_file']

VIDEO_SUFFIX = ['.MOV', '.MP4', '.AVI', '.WEBM', '.3GP', '.MKV']


def is_video_file(x: Any) -> bool:
    x = Path(x)
    cond1 = x.exists()
    cond2 = x.suffix.upper() in VIDEO_SUFFIX
    return cond1 and cond2


def video2frames(
    video_path: str,
    frame_per_sec: int = None,
) -> List[np.ndarray]:
    """
    Extracts the frames from a video using ray
    Inputs:
        video_path (str): Path to the video.
        frame_per_sec (int, Optional): The number of extracting frames per sec.
            If None, all frames will be extracted.

    Returns:
        frames (List[np.ndarray]): A list of frames.
    """
    if not is_video_file(video_path):
        raise TypeError(f'The video_path {video_path} is inappropriate.')

    # get total_frames frames of video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    # Get the original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval for frame extraction
    interval = 1 if frame_per_sec is None \
        else int(original_fps / frame_per_sec)

    frames = []
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract the frame if it's on the specified interval
        if index % interval == 0:
            frames.append(frame)
        index += 1
    cap.release()

    return frames
```

> [video2frames(video_path: str, frame_per_sec: int = None) -> List[np.ndarray]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/videotools/video2frames.py#L19)

- **說明**：從視訊中提取幀。其中可以支援的視訊格式有 `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, `.MKV`。

- **參數**
    - **video_path** (`str`)：視訊的路徑。
    - **frame_per_sec** (`int`)：每秒提取的幀數。如果為 `None`，則提取所有幀。

- **傳回值**
    - **List[np.ndarray]**：幀的列表。

- **範例**

    ```python
    import docsaidkit as D

    frames = D.video2frames('video.mp4', frame_per_sec=1)
    for frame in frames:
        D.imshow(frame)
    ```

