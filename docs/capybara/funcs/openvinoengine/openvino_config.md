# OpenVINOConfig

> [OpenVINOConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **說明**：`OpenVINOEngine` 的設定（dataclass），包含 compile/core properties、cache、request pool 等選項。

- **依賴**

  - 需要 `openvino`（請先安裝 `capybara-docsaid[openvino]`）。

- **範例**

  ```python
  from capybara.openvinoengine import OpenVINOConfig

  cfg = OpenVINOConfig(num_requests=2, copy_outputs=True)
  ```

