# OpenVINOConfig

> [OpenVINOConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **説明**：`OpenVINOEngine` の設定（dataclass）です。compile/core properties、cache、request pool などを含みます。

- **依存関係**

  - `openvino` が必要です（`capybara-docsaid[openvino]` を先にインストールしてください）。

- **例**

  ```python
  from capybara.openvinoengine import OpenVINOConfig

  cfg = OpenVINOConfig(num_requests=2, copy_outputs=True)
  ```

