# OpenVINOEngine

> [OpenVINOEngine(model_path: str | Path, device: str | OpenVINODevice = OpenVINODevice.auto, config: OpenVINOConfig | None = None, core: Any | None = None, input_shapes: dict[str, Any] | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **說明**：OpenVINO 推論封裝，提供同步推論與可選的 async queue。

- **依賴**

  - 需要 `openvino`（請先安裝 `capybara-docsaid[openvino]`）。

- **常用方法**

  - `run(feed)`：同步推論，回傳 `dict[str, np.ndarray]`。
  - `summary()`：回傳 inputs/outputs/device 等摘要。
  - `benchmark(feed, repeat=..., warmup=...)`：回傳 throughput 與 latency 統計。
  - `create_async_queue(...)`：建立 async queue（用於 pipelining）。

- **範例**

  ```python
  import numpy as np
  from capybara.openvinoengine import OpenVINOConfig, OpenVINODevice, OpenVINOEngine

  engine = OpenVINOEngine(
      "model.xml",
      device=OpenVINODevice.cpu,
      config=OpenVINOConfig(num_requests=2),
  )
  outputs = engine.run({"input": np.ones((1, 3), dtype=np.float32)})
  print(outputs.keys())
  ```

