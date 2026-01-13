# parse_metadata_from_onnx

> [parse_metadata_from_onnx(onnx_path: str | Path) -> dict[str, Any]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **Description**: Reads ONNX `custom_metadata_map` and tries to `json.loads()` string fields back to the original types.

- **Dependencies**

  - Requires `onnxruntime` (reads metadata via `InferenceSession`).

- **Parameters**

  - **onnx_path** (`str | Path`): ONNX model path.

- **Returns**

  - **dict[str, Any]**: Parsed metadata.

- **Example**

  ```python
  from capybara.onnxengine import parse_metadata_from_onnx

  meta = parse_metadata_from_onnx("model.onnx")
  print(meta)
  ```
