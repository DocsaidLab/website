# EngineConfig

> [EngineConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/engine.py)

- **Description**: High-level `ONNXEngine` configuration (dataclass) to centralize commonly used onnxruntime session/provider/run options.

- **Dependencies**

  - Requires `onnxruntime`.

- **Common fields**

  - **graph_optimization**: `"disable" | "basic" | "extended" | "all"` (or `onnxruntime.GraphOptimizationLevel`).
  - **execution_mode**: `"sequential" | "parallel"` (or `onnxruntime.ExecutionMode`).
  - **intra_op_num_threads / inter_op_num_threads**: thread counts.
  - **provider_options**: options keyed by provider name (merged into engine defaults).
  - **fallback_to_cpu**: whether to fallback to CPU when the requested provider is unavailable.
  - **enable_io_binding**: enable IO binding (may reduce copies; behavior depends on provider).
  - **enable_profiling**: enable ORT profiling.

- **Example**

  ```python
  import numpy as np
  from capybara.onnxengine import EngineConfig, ONNXEngine

  cfg = EngineConfig(
      graph_optimization="all",
      enable_io_binding=False,
      fallback_to_cpu=True,
  )
  engine = ONNXEngine("model.onnx", backend="cpu", config=cfg)

  outputs = engine.run({"input": np.ones((1, 3, 224, 224), dtype=np.float32)})
  print(outputs.keys())
  ```
