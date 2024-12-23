# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, \*\*kwargs)](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/metadata.py#L20)

- **Description**: Writes custom metadata into an ONNX model.

- **Parameters**

  - **onnx_path** (`Union[str, Path]`): The path to the ONNX model.
  - **out_path** (`Union[str, Path]`): The path to save the output ONNX model.
  - **drop_old_meta** (`bool`): Whether to remove the old metadata. Default is `False`.
  - `**kwargs`: Custom metadata.

- **Example**

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
