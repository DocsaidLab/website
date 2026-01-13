# WebDemo

> [WebDemo(camera_ip: str, color_base: str = "BGR", route: str = "/", pipelines: list | None = None) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/ipcam/app.py)

- **Dependencies**

  - Requires `flask` (install `capybara-docsaid[ipcam]`).

- **Description**: Displays an IPCam feed on a web page.

- **Parameters**

  - **camera_ip** (`str`): The URL of the IP camera.
  - **color_base** (`str`): The color space of the output frame. It can be 'BGR' or 'RGB'. Note that OpenCV input frames are always in BGR format. If `color_base` is set to 'RGB', each frame will be converted from BGR to RGB before returning. Default is 'BGR'.
  - **route** (`str`): The web page route. Default is '/'.
  - **pipelines** (`list | None`): The image processing pipeline. If `None`, it is treated as `[]`.

- **Example**

  ```python
  from capybara.vision.ipcam.app import WebDemo

  WebDemo(camera_ip='http://your_ip:your_port/video').run()
  ```
