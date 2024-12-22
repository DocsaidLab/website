# get_onnx_metadata

> [get_onnx_metadata(onnx_path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/metadata.py#L10)

- **說明**：從 ONNX 模型中取得自定義元數據。

- **參數**

  - **onnx_path** (`Union[str, Path]`)：ONNX 模型的路徑。

- **傳回值**

  - **dict**：自定義元數據。

- **範例**

  ```python
  import capybara as cb

  onnx_path = 'model.onnx'
  metadata = cb.get_onnx_metadata(onnx_path)
  print(metadata)
  # >>> metadata = {
  #     'key1': 'value1',
  #     'key2': 'value2',
  #     'key3': 'value3',
  # }
  ```
