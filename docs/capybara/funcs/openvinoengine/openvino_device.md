# OpenVINODevice

> [OpenVINODevice](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **說明**：OpenVINO device 的列舉封裝，用於 `OpenVINOEngine(device=...)`。

- **可用值**

  - `OpenVINODevice.auto` / `cpu` / `gpu` / `npu` / `hetero` / `auto_batch`

- **範例**

  ```python
  from capybara.openvinoengine import OpenVINODevice

  print(OpenVINODevice.cpu.value)  # "CPU"
  ```

