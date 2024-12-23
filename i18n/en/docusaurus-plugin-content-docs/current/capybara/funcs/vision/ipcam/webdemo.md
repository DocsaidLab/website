# WebDemo

> [WebDemo(camera_ip: str, color_base: str = 'BGR', route: str = '/', pipelines: list = []) -> None](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/app.py#L23)

- **Description**: Displays the IP camera feed on a web page.

- **Parameters**

  - **camera_ip** (`str`): The URL of the IP camera.
  - **color_base** (`str`): The color space of the output frame. It can be 'BGR' or 'RGB'. Note that OpenCV input frames are always in BGR format. If `color_base` is set to 'RGB', each frame will be converted from BGR to RGB before returning. Default is 'BGR'.
  - **route** (`str`): The web page route. Default is '/'.
  - **pipelines** (`list`): The image processing pipeline. Default is an empty list.

- **Example**

  ```python
  import capybara as cb

  cb.WebDemo(camera_ip='http://your_ip:your_port/video').run()
  ```
