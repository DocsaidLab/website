# OpenVINOEngine

> [OpenVINOEngine(model_path: str | Path, device: str | OpenVINODevice = OpenVINODevice.auto, config: OpenVINOConfig | None = None, core: Any | None = None, input_shapes: dict[str, Any] | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/openvinoengine/engine.py)

- **説明**：OpenVINO の推論 wrapper です。同期推論と（任意で）async queue を提供します。

- **依存関係**

  - `openvino` が必要です（`capybara-docsaid[openvino]` を先にインストールしてください）。

- **主なメソッド**

  - `run(feed)`: 同期推論。`dict[str, np.ndarray]` を返します。
  - `summary()`: inputs/outputs/device などの summary を返します。
  - `benchmark(feed, repeat=..., warmup=...)`: throughput / latency を計測して返します。
  - `create_async_queue(...)`: async queue を作成します（パイプライン用）。

- **例**

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

