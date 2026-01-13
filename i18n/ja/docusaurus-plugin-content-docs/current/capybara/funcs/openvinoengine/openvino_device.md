# OpenVINODevice

> [OpenVINODevice](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **説明**：OpenVINO device の enum wrapper で、`OpenVINOEngine(device=...)` から使用します。

- **利用可能な値**

  - `OpenVINODevice.auto` / `cpu` / `gpu` / `npu` / `hetero` / `auto_batch`

- **例**

  ```python
  from capybara.openvinoengine import OpenVINODevice

  print(OpenVINODevice.cpu.value)  # "CPU"
  ```

