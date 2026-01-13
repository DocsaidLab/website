# get_onnx_metadata

> [get_onnx_metadata(onnx_path: str | Path) -> dict[str, Any]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **依賴**

  - 此函式需要 `onnxruntime`（透過 InferenceSession 讀取 metadata）。

- **說明**：從 ONNX 模型中取得自定義元數據。

- **參數**

  - **onnx_path** (`Union[str, Path]`)：ONNX 模型的路徑。

- **傳回值**

  - **dict[str, Any]**：自定義元數據（原始 `custom_metadata_map` 內容，通常是字串）。

- **範例**

  ```python
  from capybara.onnxengine import get_onnx_metadata

  onnx_path = 'model.onnx'
  metadata = get_onnx_metadata(onnx_path)
  print(metadata)
  # >>> metadata = {
  #     'key1': 'value1',
  #     'key2': 'value2',
  #     'key3': 'value3',
  # }
  ```
