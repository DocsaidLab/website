---
sidebar_position: 1
---

# IpcamCapture

>[IpcamCapture(url: int, str, color_base: str) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/ipcam/camera.py#L11)

- **Description**: Capture images from an IP camera.

- **Parameters**:
    - **url** (`int`, `str`): Identifier of the video source. It can be the device index of a locally connected camera or a string containing the network address of the IP camera. For local cameras, 0 is usually the default camera. Default is 0.
    - **color_base** (`str`): Color space of the output frames. It can be 'BGR' or 'RGB'. Note that the input frames in OpenCV are always in BGR format. If color_base is set to 'RGB', each frame will be converted from BGR to RGB before returning. Default is 'BGR'.

- **Attributes**:
    - **color_base** (`str`): Color space of the output frames.

- **Methods**:
    - **get_frame() -> np.ndarray**: Retrieve the currently captured frame.

- **Example**:

    ```python
    import docsaidkit as D

    cam = D.IpcamCapture(url='http://your_ip:your_port/video')
    for frame in cam:
        D.imshow(frame)
    ```
