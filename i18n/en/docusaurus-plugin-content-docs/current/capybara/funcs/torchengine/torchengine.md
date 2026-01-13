# TorchEngine

> [TorchEngine(model_path: str | Path, device: str | Any = "cuda", output_names: Sequence[str] | None = None, config: TorchEngineConfig | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/torchengine/engine.py)

- **Description**: TorchScript inference wrapper (`torch.jit.load`) with simple input normalization and numpy outputs.

- **Dependencies**

  - Requires `torch` (install `capybara-docsaid[torchscript]` first).

- **Notes**

  - `run(feed)` builds model inputs in `feed.values()` order (current behavior).
  - If model output is a tuple/list and you specified `output_names`, its length must match the number of outputs, otherwise `ValueError` is raised.

- **Example**

  ```python
  import numpy as np
  from capybara.torchengine import TorchEngine

  engine = TorchEngine("model.pt", device="cpu")
  outputs = engine.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
  print(outputs.keys())
  ```
