# make_onnx_dynamic_axes

> [make_onnx_dynamic_axes(model_fpath: str | Path, output_fpath: str | Path, input_dims: dict[str, dict[int, str]], output_dims: dict[str, dict[int, str]], opset_version: int | None = None) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **Description**: Writes `dim_param` for specified dimensions to mark dynamic axes, and outputs a new ONNX file.

- **Dependencies**

  - Requires `onnx`.
  - If `onnxslim.simplify` is available, it will try to simplify after writing (current behavior).

- **Parameters**

  - **input_dims** / **output_dims**: Specify dynamic dims as `{tensor_name: {dim_index: dim_param}}`.
  - **opset_version**: Used to fill opset when the model has no default domain opset.

- **Limitations**

  - If the graph contains `Reshape` nodes, it raises `ValueError` (current behavior).

- **Example**

  ```python
  from capybara.onnxengine import make_onnx_dynamic_axes

  make_onnx_dynamic_axes(
      model_fpath="model.onnx",
      output_fpath="model_dynamic.onnx",
      input_dims={"input": {0: "batch"}},
      output_dims={"output": {0: "batch"}},
  )
  ```
