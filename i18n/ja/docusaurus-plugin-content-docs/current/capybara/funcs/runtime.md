# Runtime

> [Runtime / Backend](https://github.com/DocsaidLab/Capybara/blob/main/capybara/runtime.py)

- **説明**：`capybara.runtime` は `Runtime` と `Backend` のレジストリを提供し、推論 runtime と実行 backend（provider/device）の対応を定義します。

- **代表的な runtime**

  - `Runtime.onnx`: ONNXRuntime
  - `Runtime.openvino`: OpenVINO
  - `Runtime.pt`: TorchScript（PyTorch）

- **よく使う操作**

  - `Runtime.<name>.available_backends()`: runtime がサポートする backend を（定義順で）返します。
  - `Runtime.<name>.normalize_backend(backend)`: `None/str/Backend` を `Backend` に正規化します（`None` はデフォルト backend）。
  - `Runtime.<name>.auto_backend_name()`: 環境に応じて backend を選択します（例：利用可能なら CUDA を優先）。

- **備考**

  - 複数 runtime が絡む場合、`Backend.from_any(value, runtime=...)` は `runtime` 指定が必須です。指定しないと `ValueError` になります（現状の挙動）。

- **例**

  ```python
  from capybara.runtime import Runtime

  runtime = Runtime.onnx
  print([b.name for b in runtime.available_backends()])
  print(runtime.auto_backend_name())
  ```

