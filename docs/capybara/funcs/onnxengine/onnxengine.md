# ONNXEngine

> [ONNXEngine(model_path: str | Path, gpu_id: int = 0, backend: str | Backend = Backend.cpu, session_option: Mapping[str, Any] | None = None, provider_option: Mapping[str, Any] | None = None, config: EngineConfig | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/engine.py)

- **說明**：初始化 ONNX 模型推論引擎。

- **依賴**

  - 此模組依賴 `onnxruntime`（CPU 或 GPU 版），請先安裝：

    ```bash
    pip install "capybara-docsaid[onnxruntime]"      # CPU
    # or
    pip install "capybara-docsaid[onnxruntime-gpu]"  # GPU
    ```

- **參數**

  - **model_path** (`str | Path`)：ONNX 模型檔案路徑。
  - **gpu_id** (`int`)：GPU ID。預設為 0。
  - **backend** (`str | Backend`)：推論後端（runtime=`onnx`）。常用值：
    - `Backend.cpu`
    - `Backend.cuda`
    - `Backend.tensorrt`
    - `Backend.tensorrt_rtx`
  - **session_option** (`Mapping[str, Any] | None`)：SessionOptions 的覆寫參數（會以 `setattr` 或 `add_session_config_entry` 方式套用）。
  - **provider_option** (`Mapping[str, Any] | None`)：Execution Provider 的覆寫參數（例如 `device_id`、cache 等）。
  - **config** (`EngineConfig | None`)：較高階的推論設定（graph optimization、threading、IOBinding、profiling 等）。

- **推論**

  - `engine.run(feed)`：以 `Mapping[str, np.ndarray]` 推論。
  - `engine(**inputs)`：以 keyword arguments 推論（若只傳一個 `dict` 也會被視為 feed）。
  - 回傳值為 `dict[str, np.ndarray]`，key 為模型 output name。

- **常用方法**

  - `summary()`：回傳 inputs/outputs/providers 等摘要資訊。
  - `benchmark(inputs, repeat=100, warmup=10)`：回傳 throughput 與 latency 統計。

- **範例**

  ```python
  import numpy as np
  from capybara.onnxengine import EngineConfig, ONNXEngine
  from capybara.runtime import Runtime

  model_path = 'model.onnx'
  runtime = Runtime.onnx

  engine = ONNXEngine(
      model_path,
      backend=runtime.auto_backend_name(),
      config=EngineConfig(enable_io_binding=False),
  )

  inputs = {
      'input': np.random.randn(1, 3, 224, 224).astype(np.float32),
  }
  outputs = engine.run(inputs)
  print(outputs.keys())
  print(engine.summary())
  ```
