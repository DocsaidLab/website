# get_onnx_output_infos

> [get_onnx_output_infos(model: str | Path | onnx.ModelProto) -> dict[str, dict[str, Any]]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **Description**: Returns output shape and dtype info of an ONNX model.

- **Dependencies**

  - Requires `onnx`.

- **Example**

  ```python
  from capybara.onnxengine import get_onnx_output_infos

  infos = get_onnx_output_infos("model.onnx")
  print(infos)
  ```
