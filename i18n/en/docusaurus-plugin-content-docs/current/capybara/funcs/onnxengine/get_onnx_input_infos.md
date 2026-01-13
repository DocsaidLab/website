# get_onnx_input_infos

> [get_onnx_input_infos(model: str | Path | onnx.ModelProto) -> dict[str, dict[str, Any]]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **Description**: Returns input shape and dtype info of an ONNX model.

- **Dependencies**

  - Requires `onnx`.

- **Return format**

  - Returns `{input_name: {"shape": [...], "dtype": np.dtype}}`.
  - Shape dims may be represented differently depending on the source:
    - When `dim_param` is present: returns a string (e.g. `"batch"`).
    - When `dim_value == 0` or unknown: returns `-1`.

- **Example**

  ```python
  from capybara.onnxengine import get_onnx_input_infos

  infos = get_onnx_input_infos("model.onnx")
  print(infos)
  ```
