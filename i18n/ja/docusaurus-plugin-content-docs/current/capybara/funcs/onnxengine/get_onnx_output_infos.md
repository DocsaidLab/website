# get_onnx_output_infos

> [get_onnx_output_infos(model: str | Path | onnx.ModelProto) -> dict[str, dict[str, Any]]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **説明**：ONNX モデルの出力 tensor の shape / dtype 情報を返します。

- **依存関係**

  - `onnx` が必要です。

- **例**

  ```python
  from capybara.onnxengine import get_onnx_output_infos

  infos = get_onnx_output_infos("model.onnx")
  print(infos)
  ```

