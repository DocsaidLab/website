# make_onnx_dynamic_axes

> [make_onnx_dynamic_axes(model_fpath: str | Path, output_fpath: str | Path, input_dims: dict[str, dict[int, str]], output_dims: dict[str, dict[int, str]], opset_version: int | None = None) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/utils.py)

- **説明**：指定次元に `dim_param` を書き込み、dynamic axis をマークした新しい ONNX ファイルを出力します。

- **依存関係**

  - `onnx` が必要です。
  - `onnxslim.simplify` が利用可能な場合、書き込み後に simplify を試みます（現状の挙動）。

- **パラメータ**

  - **input_dims** / **output_dims**：`{tensor_name: {dim_index: dim_param}}` の形式で dynamic dims を指定します。
  - **opset_version**：モデルに default domain の opset がない場合に opset を補完するために使用します。

- **制約**

  - グラフ内に `Reshape` ノードが含まれる場合、`ValueError` を raise します（現状の挙動）。

- **例**

  ```python
  from capybara.onnxengine import make_onnx_dynamic_axes

  make_onnx_dynamic_axes(
      model_fpath="model.onnx",
      output_fpath="model_dynamic.onnx",
      input_dims={"input": {0: "batch"}},
      output_dims={"output": {0: "batch"}},
  )
  ```

