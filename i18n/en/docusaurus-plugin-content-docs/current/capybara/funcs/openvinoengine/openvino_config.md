# OpenVINOConfig

> [OpenVINOConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **Description**: `OpenVINOEngine` configuration (dataclass), including compile/core properties, cache, request pool, etc.

- **Dependencies**

  - Requires `openvino` (install `capybara-docsaid[openvino]` first).

- **Example**

  ```python
  from capybara.openvinoengine import OpenVINOConfig

  cfg = OpenVINOConfig(num_requests=2, copy_outputs=True)
  ```
