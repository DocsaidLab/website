# get_onnx_metadata

> [get_onnx_metadata(onnx_path: str | Path) -> dict[str, Any]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **依存関係**

  - `onnxruntime` が必要です（InferenceSession 経由で metadata を読み取ります）。

- **説明**：ONNX モデルから custom metadata を取得します。

- **パラメータ**

  - **onnx_path** (`str | Path`)：ONNX モデルのパス。

- **戻り値**

  - **dict[str, Any]**：custom metadata（`custom_metadata_map` の内容。通常は文字列）。

- **例**

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

