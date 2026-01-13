# IpcamCapture

> [IpcamCapture(url: int | str = 0, color_base: str = "BGR") -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/ipcam/camera.py)

- **Dependencies**

  - `ipcam` is an optional extra. If you want to use `WebDemo`, install first:

    ```bash
    pip install "capybara-docsaid[ipcam]"
    ```

- **Description**: Captures frames from a camera source (local camera index or IPCam URL).

- **Parameters**

  - **url** (`int`, `str`): The identifier for the video source. It can be the device index for a locally connected camera or a string containing the network address of an IP camera. For local cameras, 0 is typically the default camera. Default is 0.
  - **color_base** (`str`): The color space for the output frame. It can be 'BGR' or 'RGB'. Note that OpenCV input frames are always in BGR format. If `color_base` is set to 'RGB', each frame will be converted from BGR to RGB before returning. Default is 'BGR'.

- **Attributes**

  - **color_base** (`str`): The color space of the output frame.

- **Methods**

  - **get_frame() -> np.ndarray**: Retrieves the current captured frame.

- **Example**

  ```python
  from capybara import imwrite
  from capybara.vision.ipcam.camera import IpcamCapture

  cam = IpcamCapture(url='http://your_ip:your_port/video')
  for frame in cam:
      imwrite(frame, 'frame.jpg')
  ```
