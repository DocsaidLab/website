# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, \*\*kwargs)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/metadata.py#L20)

- **說明**：將自定義元數據寫入 ONNX 模型中。

- **參數**

  - **onnx_path** (`Union[str, Path]`)：ONNX 模型的路徑。
  - **out_path** (`Union[str, Path]`)：輸出 ONNX 模型的路徑。
  - **drop_old_meta** (`bool`)：是否刪除舊的元數據。預設為 `False`。
  - `**kwargs`：自定義元數據。

- **範例**

  ```python
  import capybara as cb

  onnx_path = 'model.onnx'
  out_path = 'model_with_metadata.onnx'
  cb.write_metadata_into_onnx(
      onnx_path,
      out_path,
      drop_old_meta=False,
      key1='value1',
      key2='value2',
      key3='value3',
  )
  ```
