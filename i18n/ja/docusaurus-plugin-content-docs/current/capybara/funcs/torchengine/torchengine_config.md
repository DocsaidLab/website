# TorchEngineConfig

> [TorchEngineConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/torchengine/engine.py)

- **説明**：`TorchEngine` の設定（dataclass）です。現状は `dtype` と `cuda_sync` を含みます。

- **依存関係**

  - `torch` が必要です（`capybara-docsaid[torchscript]` を先にインストールしてください）。

- **例**

  ```python
  from capybara.torchengine import TorchEngineConfig

  cfg = TorchEngineConfig(dtype="fp16", cuda_sync=True)
  ```

