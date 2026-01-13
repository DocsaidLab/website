# get_onnx_metadata

> [get_onnx_metadata(onnx_path: str | Path) -> dict[str, Any]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **Dependencies**

  - Requires `onnxruntime` (reads metadata via `InferenceSession`).

- **Description**: Retrieves custom metadata from an ONNX model.

- **Parameters**

  - **onnx_path** (`str | Path`): Path to the ONNX model.

- **Returns**

  - **dict[str, Any]**: Raw `custom_metadata_map` content (typically strings).

- **Example**

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
