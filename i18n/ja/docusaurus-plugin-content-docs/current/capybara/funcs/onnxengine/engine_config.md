# EngineConfig

> [EngineConfig](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/engine.py)

- **説明**：`ONNXEngine` の高レベル設定（dataclass）で、onnxruntime の session/provider/run オプションをまとめて管理します。

- **依存関係**

  - `onnxruntime` が必要です。

- **主なフィールド**

  - **graph_optimization**: `"disable" | "basic" | "extended" | "all"`（または `onnxruntime.GraphOptimizationLevel`）。
  - **execution_mode**: `"sequential" | "parallel"`（または `onnxruntime.ExecutionMode`）。
  - **intra_op_num_threads / inter_op_num_threads**: スレッド数。
  - **provider_options**: provider 名をキーとする options（engine のデフォルトとマージされます）。
  - **fallback_to_cpu**: 指定 provider が使えない場合に CPU へフォールバックするかどうか。
  - **enable_io_binding**: IO binding を有効にします（コピー削減が期待できますが、挙動は provider に依存します）。
  - **enable_profiling**: ORT profiling を有効にします。

- **例**

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

