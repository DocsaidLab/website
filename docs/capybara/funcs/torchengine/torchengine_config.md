# TorchEngineConfig

> [TorchEngineConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/torchengine/engine.py)

- **說明**：`TorchEngine` 的設定（dataclass），目前提供 `dtype` 與 `cuda_sync`。

- **依賴**

  - 需要 `torch`（請先安裝 `capybara-docsaid[torchscript]`）。

- **範例**

  ```python
  from capybara.torchengine import TorchEngineConfig

  cfg = TorchEngineConfig(dtype="fp16", cuda_sync=True)
  ```

