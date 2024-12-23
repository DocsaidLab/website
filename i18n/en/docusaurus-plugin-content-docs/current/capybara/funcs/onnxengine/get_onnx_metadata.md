# get_onnx_metadata

> [get_onnx_metadata(onnx_path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/metadata.py#L10)

- **Description**: Retrieves custom metadata from an ONNX model.

- **Parameters**

  - **onnx_path** (`Union[str, Path]`): The path to the ONNX model.

- **Returns**

  - **dict**: Custom metadata.

- **Example**

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
