# TorchEngine

> [TorchEngine(model_path: str | Path, device: str | Any = "cuda", output_names: Sequence[str] | None = None, config: TorchEngineConfig | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/torchengine/engine.py)

- **說明**：TorchScript 推論封裝（`torch.jit.load`），提供簡單的輸入正規化與輸出轉為 numpy。

- **依賴**

  - 需要 `torch`（請先安裝 `capybara-docsaid[torchscript]`）。

- **注意**

  - `run(feed)` 會依 `feed.values()` 的順序組成輸入（以目前實作為準）。
  - 若 model 輸出為 tuple/list，且你有指定 `output_names`，其長度必須與輸出數量一致，否則會拋出 `ValueError`。

- **範例**

  ```python
  import numpy as np
  from capybara.torchengine import TorchEngine

  engine = TorchEngine("model.pt", device="cpu")
  outputs = engine.run({"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
  print(outputs.keys())
  ```

