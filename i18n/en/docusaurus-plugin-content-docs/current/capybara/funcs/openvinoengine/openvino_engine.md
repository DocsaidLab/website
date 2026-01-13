# OpenVINOEngine

> [OpenVINOEngine(model_path: str | Path, device: str | OpenVINODevice = OpenVINODevice.auto, config: OpenVINOConfig | None = None, core: Any | None = None, input_shapes: dict[str, Any] | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **Description**: OpenVINO inference wrapper that provides sync inference and an optional async queue.

- **Dependencies**

  - Requires `openvino` (install `capybara-docsaid[openvino]` first).

- **Common methods**

  - `run(feed)`: Sync inference, returns `dict[str, np.ndarray]`.
  - `summary()`: Returns a summary of inputs/outputs/device, etc.
  - `benchmark(feed, repeat=..., warmup=...)`: Returns throughput and latency stats.
  - `create_async_queue(...)`: Creates an async queue (for pipelining).

- **Example**

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
