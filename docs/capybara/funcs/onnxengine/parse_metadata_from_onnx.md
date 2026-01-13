# parse_metadata_from_onnx

> [parse_metadata_from_onnx(onnx_path: str | Path) -> dict[str, Any]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **說明**：讀取 ONNX 的 `custom_metadata_map`，並嘗試將字串欄位以 `json.loads()` 解析回原始型別。

- **依賴**

  - 需要 `onnxruntime`（透過 InferenceSession 讀取 metadata）。

- **參數**

  - **onnx_path** (`str | Path`)：ONNX 模型路徑。

- **傳回值**

  - **dict[str, Any]**：解析後的 metadata。

- **範例**

  ```python
  from capybara.onnxengine import parse_metadata_from_onnx

  meta = parse_metadata_from_onnx("model.onnx")
  print(meta)
  ```

