# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: str | Path, out_path: str | Path, drop_old_meta: bool = False, **kwargs: Any) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **Dependencies**

  - Requires `onnx` (read/write model files).
  - When `drop_old_meta=False`, it reads existing metadata and also requires `onnxruntime`.

- **Description**: Writes custom metadata into an ONNX model.

- **Parameters**

  - **onnx_path** (`str | Path`): Path to the input ONNX model.
  - **out_path** (`str | Path`): Output ONNX model path.
  - **drop_old_meta** (`bool`): Whether to drop existing metadata. Default is `False`.
  - `**kwargs`: Custom metadata fields.

- **Behavior**

  - Automatically adds `Date` (via `capybara.utils.time.now(fmt=...)`).
  - Each metadata value is serialized to string via `json.dumps()` and written into ONNX props.

- **Example**

  ```python
  from capybara.onnxengine import write_metadata_into_onnx

  onnx_path = 'model.onnx'
  out_path = 'model_with_metadata.onnx'
  write_metadata_into_onnx(
      onnx_path,
      out_path,
      drop_old_meta=False,
      key1='value1',
      key2='value2',
      key3='value3',
  )
  ```
