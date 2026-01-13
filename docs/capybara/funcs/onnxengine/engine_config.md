# EngineConfig

> [EngineConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/engine.py)

- **說明**：`ONNXEngine` 的高階設定（dataclass），用來集中管理 onnxruntime session / provider / run 的常用選項。

- **依賴**

  - 需要 `onnxruntime`。

- **常用欄位**

  - **graph_optimization**：`"disable" | "basic" | "extended" | "all"`（或直接給 `onnxruntime.GraphOptimizationLevel`）。
  - **execution_mode**：`"sequential" | "parallel"`（或直接給 `onnxruntime.ExecutionMode`）。
  - **intra_op_num_threads / inter_op_num_threads**：執行緒數。
  - **provider_options**：以 provider name 為 key 的 options（會與引擎預設值合併）。
  - **fallback_to_cpu**：當指定 provider 不可用時是否回退到 CPU。
  - **enable_io_binding**：是否使用 IO binding（可減少部分 copy，但行為依 provider 而異）。
  - **enable_profiling**：啟用 ORT profiling。

- **範例**

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

