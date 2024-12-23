# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, \*\*kwargs)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/metadata.py#L20)

- **説明**：カスタムメタデータを ONNX モデルに書き込みます。

- **パラメータ**

  - **onnx_path** (`Union[str, Path]`)：ONNX モデルのパス。
  - **out_path** (`Union[str, Path]`)：出力先 ONNX モデルのパス。
  - **drop_old_meta** (`bool`)：古いメタデータを削除するかどうか。デフォルトは `False`。
  - `**kwargs`：カスタムメタデータ。

- **例**

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
