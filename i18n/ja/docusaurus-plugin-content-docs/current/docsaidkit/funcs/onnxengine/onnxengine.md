---
sidebar_position: 1
---

# ONNXEngine

> [ONNXEngine(model_path: Union[str, Path], gpu_id: int = 0, backend: Union[str, int, Backend] = Backend.cpu, session_option: Dict[str, Any] = {}, provider_option: Dict[str, Any] = {})](https://github.com/DocsaidLab/DocsaidKit/blob/main/docsaidkit/onnxengine/engine.py)

- **説明**：ONNX モデル推論エンジンを初期化します。

- **パラメータ**

  - **model_path** (`Union[str, Path]`)：ファイル名またはシリアライズされた ONNX または ORT 形式のモデルのバイト列。
  - **gpu_id** (`int`)：GPU ID。デフォルトは 0。
  - **backend** (`Union[str, int, Backend]`)：推論バックエンド、`Backend.cpu` または `Backend.cuda` を選択できます。デフォルトは `Backend.cpu`。
  - **session_option** (`Dict[str, Any]`)：onnxruntime.SessionOptions のパラメータで、セッションオプションを設定します。デフォルトは `{}`。設定方法の詳細については、[SessionOptions](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions) を参照してください。
  - **provider_option** (`Dict[str, Any]`)：onnxruntime.provider_options のパラメータ。デフォルトは `{}`。詳細設定方法については、[CUDAExecutionProvider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options) を参照してください。

- **推論**

  モデルをロードすると、ONNX ファイル内の情報が読み込まれ、入力と出力に関する形状やデータ型を含む辞書が設定されます。これにより、`ONNXEngine` インスタンスを呼び出すと、直接その辞書を使って出力結果を得ることができます。

- **例**

  ```python
  import docsaidkit as D

  model_path = 'model.onnx'
  engine = D.ONNXEngine(model_path)
  print(engine)

  # 推論
  # モデルに2つの入力と2つの出力があり、名前は以下の通りだと仮定します：
  #   'input1', 'input2', 'output1', 'output2'。
  input_data = {
      'input1': np.random.randn(1, 3, 224, 224).astype(np.float32),
      'input2': np.random.randn(1, 3, 224, 224).astype(np.float32),
  }

  outputs = engine(**input_data)

  output_data1 = outputs['output1']
  output_data2 = outputs['output2']
  ```
