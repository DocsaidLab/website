# Runtime

> [Runtime / Backend](https://github.com/DocsaidLab/Capybara/blob/main/capybara/runtime.py)

- **說明**：`capybara.runtime` 提供 `Runtime` 與 `Backend` 的註冊表，用來描述「推論 runtime」與「實際執行後端（provider/device）」的對應關係。

- **常見 Runtime**

  - `Runtime.onnx`：ONNXRuntime
  - `Runtime.openvino`：OpenVINO
  - `Runtime.pt`：TorchScript（PyTorch）

- **常用操作**

  - `Runtime.<name>.available_backends()`：取得該 runtime 支援的後端列表（依定義順序）。
  - `Runtime.<name>.normalize_backend(backend)`：把 `None/str/Backend` 正規化成 `Backend`（`None` 會使用 default backend）。
  - `Runtime.<name>.auto_backend_name()`：依環境自動選擇後端（例如優先使用 CUDA）。

- **注意**

  - `Backend.from_any(value, runtime=...)` 在多 runtime 共存時需要指定 `runtime`，否則會拋出 `ValueError`（以目前實作為準）。

- **範例**

  ```python
  from capybara.runtime import Runtime

  runtime = Runtime.onnx
  print([b.name for b in runtime.available_backends()])
  print(runtime.auto_backend_name())
  ```

