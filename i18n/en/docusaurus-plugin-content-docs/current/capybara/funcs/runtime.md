# Runtime

> [Runtime / Backend](https://github.com/DocsaidLab/Capybara/blob/main/capybara/runtime.py)

- **Description**: `capybara.runtime` provides registries of `Runtime` and `Backend`, describing the mapping between an inference runtime and the actual execution backend (provider/device).

- **Common runtimes**

  - `Runtime.onnx`: ONNXRuntime
  - `Runtime.openvino`: OpenVINO
  - `Runtime.pt`: TorchScript (PyTorch)

- **Common operations**

  - `Runtime.<name>.available_backends()`: Returns the supported backends of the runtime (in defined order).
  - `Runtime.<name>.normalize_backend(backend)`: Normalizes `None/str/Backend` to `Backend` (`None` uses the default backend).
  - `Runtime.<name>.auto_backend_name()`: Chooses a backend based on the environment (e.g. prefer CUDA when available).

- **Notes**

  - When multiple runtimes are involved, `Backend.from_any(value, runtime=...)` requires `runtime`; otherwise it raises `ValueError` (current behavior).

- **Example**

  ```python
  from capybara.runtime import Runtime

  runtime = Runtime.onnx
  print([b.name for b in runtime.available_backends()])
  print(runtime.auto_backend_name())
  ```
