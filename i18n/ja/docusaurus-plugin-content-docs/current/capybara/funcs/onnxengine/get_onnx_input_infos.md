# get_onnx_input_infos

> [get_onnx_input_infos(model: str | Path | onnx.ModelProto) -> dict[str, dict[str, Any]]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **説明**：ONNX モデルの入力 tensor の shape / dtype 情報を返します。

- **依存関係**

  - `onnx` が必要です。

- **返り値の形式**

  - `{input_name: {"shape": [...], "dtype": np.dtype}}` を返します。
  - shape の各次元はソースによって表現が異なる場合があります：
    - `dim_param` がある場合：文字列（例：`"batch"`）になります。
    - `dim_value == 0` または不明な場合：`-1` になります。

- **例**

  ```python
  from capybara.onnxengine import get_onnx_input_infos

  infos = get_onnx_input_infos("model.onnx")
  print(infos)
  ```

