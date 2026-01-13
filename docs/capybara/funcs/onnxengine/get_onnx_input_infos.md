# get_onnx_input_infos

> [get_onnx_input_infos(model: str | Path | onnx.ModelProto) -> dict[str, dict[str, Any]]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **說明**：取得 ONNX model inputs 的 shape 與 dtype 資訊。

- **依賴**

  - 需要 `onnx`。

- **傳回值格式**

  - 回傳值為 `{input_name: {"shape": [...], "dtype": np.dtype}}`。
  - shape 維度會依來源以不同型別表示：
    - `dim_param` 存在時：回傳字串（例如 `"batch"`）。
    - `dim_value == 0` 或未知：回傳 `-1`。

- **範例**

  ```python
  from capybara.onnxengine import get_onnx_input_infos

  infos = get_onnx_input_infos("model.onnx")
  print(infos)
  ```

