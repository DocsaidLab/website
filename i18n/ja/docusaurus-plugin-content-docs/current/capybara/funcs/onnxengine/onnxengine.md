# ONNXEngine

> [ONNXEngine(model_path: Union[str, Path], gpu_id: int = 0, backend: Union[str, int, Backend] = Backend.cpu, session_option: Dict[str, Any] = {}, provider_option: Dict[str, Any] = {})](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/engine.py#L18)

- **説明**：ONNX モデル推論エンジンを初期化します。

- **パラメータ**

  - **model_path** (`Union[str, Path]`)：ファイル名またはシリアライズされた ONNX または ORT 形式のモデルのバイト文字列。
  - **gpu_id** (`int`)：GPU ID。デフォルトは 0。
  - **backend** (`Union[str, int, Backend]`)：推論バックエンド。`Backend.cpu` または `Backend.cuda` を選択可能。デフォルトは `Backend.cpu`。
  - **session_option** (`Dict[str, Any]`)：onnxruntime.SessionOptions のパラメータで、セッションオプションを設定します。デフォルトは `{}`。詳細設定については：[SessionOptions](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions) を参照してください。
  - **provider_option** (`Dict[str, Any]`)：onnxruntime.provider_options のパラメータ。デフォルトは `{}`。詳細設定については：[CUDAExecutionProvider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options) を参照してください。

- **推論**

  モデルをロードする際に、関数は ONNX ファイル内の情報を読み込み、入力と出力値に対して、形状とデータ型を含む辞書を設定します。

  したがって、`ONNXEngine` インスタンスを呼び出すと、その辞書を使って出力結果を直接取得できます。

- **例**

  ```python
  import capybara as cb

  model_path = 'model.onnx'
  engine = cb.ONNXEngine(model_path)
  print(engine)

  # 推論
  # モデルには2つの入力と2つの出力があり、それぞれ名前が付けられていると仮定：
  #   'input1', 'input2', 'output1', 'output2'。
  input_data = {
      'input1': np.random.randn(1, 3, 224, 224).astype(np.float32),
      'input2': np.random.randn(1, 3, 224, 224).astype(np.float32),
  }

  outputs = engine(**input_data)

  output_data1 = outputs['output1']
  output_data2 = outputs['output2']
  ```
