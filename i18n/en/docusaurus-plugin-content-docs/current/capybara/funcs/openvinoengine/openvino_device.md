# OpenVINODevice

> [OpenVINODevice](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **Description**: Enum wrapper for OpenVINO devices, used by `OpenVINOEngine(device=...)`.

- **Available values**

  - `OpenVINODevice.auto` / `cpu` / `gpu` / `npu` / `hetero` / `auto_batch`

- **Example**

  ```python
  from capybara.openvinoengine import OpenVINODevice

  print(OpenVINODevice.cpu.value)  # "CPU"
  ```
