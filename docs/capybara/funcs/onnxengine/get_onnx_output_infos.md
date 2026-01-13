# get_onnx_output_infos

> [get_onnx_output_infos(model: str | Path | onnx.ModelProto) -> dict[str, dict[str, Any]]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **說明**：取得 ONNX model outputs 的 shape 與 dtype 資訊。

- **依賴**

  - 需要 `onnx`。

- **範例**

  ```python
  from capybara.onnxengine import get_onnx_output_infos

  infos = get_onnx_output_infos("model.onnx")
  print(infos)
  ```

