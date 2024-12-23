# IpcamCapture

> [IpcamCapture(url: int, str, color_base: str) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/camera.py#L11)

- **Description**: Captures images from an IP camera.

- **Parameters**

  - **url** (`int`, `str`): The identifier for the video source. It can be the device index for a locally connected camera or a string containing the network address of an IP camera. For local cameras, 0 is typically the default camera. Default is 0.
  - **color_base** (`str`): The color space for the output frame. It can be 'BGR' or 'RGB'. Note that OpenCV input frames are always in BGR format. If `color_base` is set to 'RGB', each frame will be converted from BGR to RGB before returning. Default is 'BGR'.

- **Attributes**

  - **color_base** (`str`): The color space of the output frame.

- **Methods**

  - **get_frame() -> np.ndarray**: Retrieves the current captured frame.

- **Example**

  ```python
  import capybara as cb

  cam = cb.IpcamCapture(url='http://your_ip:your_port/video')
  for frame in cam:
      cb.imwrite(frame, 'frame.jpg')
  ```
