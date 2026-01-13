# ONNXEngine

> [ONNXEngine(model_path: str | Path, gpu_id: int = 0, backend: str | Backend = Backend.cpu, session_option: Mapping[str, Any] | None = None, provider_option: Mapping[str, Any] | None = None, config: EngineConfig | None = None)](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/engine.py)

- **説明**：ONNX モデル推論エンジンを初期化します。

- **依存関係**

  - 本モジュールは `onnxruntime`（CPU または GPU 版）に依存します。先にインストールしてください：

    ```bash
    pip install "capybara-docsaid[onnxruntime]"      # CPU
    # or
    pip install "capybara-docsaid[onnxruntime-gpu]"  # GPU
    ```

- **パラメータ**

  - **model_path** (`str | Path`)：ONNX モデルファイルのパス。
  - **gpu_id** (`int`)：GPU ID。デフォルトは 0。
  - **backend** (`str | Backend`)：推論 backend（runtime=`onnx`）。代表的な値：
    - `Backend.cpu`
    - `Backend.cuda`
    - `Backend.tensorrt`
    - `Backend.tensorrt_rtx`
  - **session_option** (`Mapping[str, Any] | None`)：SessionOptions の上書きパラメータ（`setattr` / `add_session_config_entry` で適用）。
  - **provider_option** (`Mapping[str, Any] | None`)：Execution Provider の上書きパラメータ（例：`device_id` や cache 等）。
  - **config** (`EngineConfig | None`)：高レベル推論設定（graph optimization、threading、IOBinding、profiling 等）。

- **推論**

  - `engine.run(feed)`：`Mapping[str, np.ndarray]` で推論します。
  - `engine(**inputs)`：keyword arguments で推論します（`dict` を 1 つ渡した場合も feed として扱われます）。
  - 返り値は `dict[str, np.ndarray]` で、key はモデルの output name です。

- **主なメソッド**

  - `summary()`：inputs/outputs/providers などの概要を返します。
  - `benchmark(inputs, repeat=100, warmup=10)`：throughput と latency の統計を返します。

- **例**

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

