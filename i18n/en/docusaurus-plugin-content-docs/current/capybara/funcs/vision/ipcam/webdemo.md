---
sidebar_position: 2
---

# WebDemo

> [WebDemo(camera_ip: str, color_base: str = 'BGR', route: str = '/', pipelines: list = []) -> None](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/vision/ipcam/app.py#L23)

- **Description**: Display images from an IP camera on a web page.

- **Parameters**:
    - **camera_ip** (`str`): URL of the IP camera.
    - **color_base** (`str`): Color space of the output frames. It can be 'BGR' or 'RGB'. Note that the input frames in OpenCV are always in BGR format. If color_base is set to 'RGB', each frame will be converted from BGR to RGB before returning. Default is 'BGR'.
    - **route** (`str`): Web page route. Default is '/'.
    - **pipelines** (`list`): Image processing pipelines. Default is an empty list.

- **Example**:

    ```python
    import docsaidkit as D

    D.WebDemo(camera_ip='http://your_ip:your_port/video').run()
    ```
