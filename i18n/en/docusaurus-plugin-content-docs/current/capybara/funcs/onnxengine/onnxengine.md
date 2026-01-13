# ONNXEngine

> [ONNXEngine(model_path: str | Path, gpu_id: int = 0, backend: str | Backend = Backend.cpu, session_option: Mapping[str, Any] | None = None, provider_option: Mapping[str, Any] | None = None, config: EngineConfig | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/engine.py)

- **Description**: Initializes an ONNX model inference engine.

- **Dependencies**

  - This module depends on `onnxruntime` (CPU or GPU). Install one of:

    ```bash
    pip install "capybara-docsaid[onnxruntime]"      # CPU
    # or
    pip install "capybara-docsaid[onnxruntime-gpu]"  # GPU
    ```

- **Parameters**

  - **model_path** (`str | Path`): Path to the ONNX model file.
  - **gpu_id** (`int`): GPU id. Default is 0.
  - **backend** (`str | Backend`): Inference backend (runtime=`onnx`). Common values:
    - `Backend.cpu`
    - `Backend.cuda`
    - `Backend.tensorrt`
    - `Backend.tensorrt_rtx`
  - **session_option** (`Mapping[str, Any] | None`): Overrides for `SessionOptions` (applied via `setattr` or `add_session_config_entry`).
  - **provider_option** (`Mapping[str, Any] | None`): Overrides for Execution Provider options (e.g. `device_id`, cache, etc.).
  - **config** (`EngineConfig | None`): Higher-level inference settings (graph optimization, threading, IO binding, profiling, etc.).

- **Inference**

  - `engine.run(feed)`: Runs with `Mapping[str, np.ndarray]`.
  - `engine(**inputs)`: Runs with keyword arguments (passing a single `dict` is treated as feed).
  - Returns `dict[str, np.ndarray]`, keyed by output names.

- **Common methods**

  - `summary()`: Returns a summary of inputs/outputs/providers, etc.
  - `benchmark(inputs, repeat=100, warmup=10)`: Returns throughput/latency stats.

- **Example**

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
