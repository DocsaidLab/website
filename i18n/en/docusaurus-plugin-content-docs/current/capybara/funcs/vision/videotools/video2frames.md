---
sidebar_position: 1
---

# video2frames

> [video2frames(video_path: str, frame_per_sec: int = None) -> List[np.ndarray]](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/videotools/video2frames.py#L19)

- **Description**: Extract frames from a video. Supported video formats include `.MOV`, `.MP4`, `.AVI`, `.WEBM`, `.3GP`, `.MKV`.

- **Parameters**:
    - **video_path** (`str`): Path to the video.
    - **frame_per_sec** (`int`): Number of frames to extract per second. If `None`, extract all frames.

- **Returns**:
    - **List[np.ndarray]**: List of frames.

- **Example**:

    ```python
    import docsaidkit as D

    frames = D.video2frames('video.mp4', frame_per_sec=1)
    for frame in frames:
        D.imshow(frame)
    ```
