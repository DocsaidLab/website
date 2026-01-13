# make_onnx_dynamic_axes

> [make_onnx_dynamic_axes(model_fpath: str | Path, output_fpath: str | Path, input_dims: dict[str, dict[int, str]], output_dims: dict[str, dict[int, str]], opset_version: int | None = None) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **說明**：將 ONNX model 的指定維度寫入 `dim_param`，用於標記 dynamic axes，並輸出新的 ONNX 檔案。

- **依賴**

  - 需要 `onnx`。
  - 若環境提供 `onnxslim.simplify`，會在寫入後嘗試 simplify（以目前實作為準）。

- **參數**

  - **input_dims** / **output_dims**：以 `{tensor_name: {dim_index: dim_param}}` 指定要標記為 dynamic 的維度。
  - **opset_version**：當原 model 沒有 default domain opset 時，用於補上 opset。

- **限制**

  - 若 graph 內含 `Reshape` 節點，會直接拋出 `ValueError`（以目前實作為準）。

- **範例**

  ```python
  from capybara.onnxengine import make_onnx_dynamic_axes

  make_onnx_dynamic_axes(
      model_fpath="model.onnx",
      output_fpath="model_dynamic.onnx",
      input_dims={"input": {0: "batch"}},
      output_dims={"output": {0: "batch"}},
  )
  ```

