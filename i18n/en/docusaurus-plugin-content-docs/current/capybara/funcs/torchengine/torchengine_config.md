# TorchEngineConfig

> [TorchEngineConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/torchengine/engine.py)

- **Description**: `TorchEngine` configuration (dataclass). Currently includes `dtype` and `cuda_sync`.

- **Dependencies**

  - Requires `torch` (install `capybara-docsaid[torchscript]` first).

- **Example**

  ```python
  from capybara.torchengine import TorchEngineConfig

  cfg = TorchEngineConfig(dtype="fp16", cuda_sync=True)
  ```
