# TorchEngine

> [TorchEngine(model_path: str | Path, device: str | Any = "cuda", output_names: Sequence[str] | None = None, config: TorchEngineConfig | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/torchengine/engine.py)

- **説明**：TorchScript 推論 wrapper（`torch.jit.load`）で、入力の簡易正規化と NumPy 出力を提供します。

- **依存関係**

  - `torch` が必要です（`capybara-docsaid[torchscript]` を先にインストールしてください）。

- **備考**

  - `run(feed)` は `feed.values()` の順序で model 入力を構築します（現状の挙動）。
  - 出力が tuple/list のモデルで `output_names` を指定する場合、出力数と長さが一致しないと `ValueError` になります。

- **例**

  ```python
  import numpy as np
  from capybara.torchengine import TorchEngine

  engine = TorchEngine("model.pt", device="cpu")
  outputs = engine.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
  print(outputs.keys())
  ```

