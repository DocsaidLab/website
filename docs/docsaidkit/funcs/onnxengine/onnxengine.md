---
sidebar_position: 1
---

# ONNXEngine

>[ONNXEngine(model_path: Union[str, Path], gpu_id: int = 0, backend: Union[str, int, Backend] = Backend.cpu, session_option: Dict[str, Any] = {}, provider_option: Dict[str, Any] = {})](https://github.com/DocsaidLab/DocsaidKit/blob/main/docsaidkit/onnxengine/engine.py)

- **說明**：初始化 ONNX 模型推論引擎。

- **參數**

    - **model_path** (`Union[str, Path]`)：檔案名稱或序列化的 ONNX 或 ORT 格式模型的位元組字串。
    - **gpu_id** (`int`)：GPU ID。預設為 0。
    - **backend** (`Union[str, int, Backend]`)：推論後端，可以選 `Backend.cpu` 或 `Backend.cuda`。預設為 `Backend.cpu`。
    - **session_option** (`Dict[str, Any]`)：這是 onnxruntime.SessionOptions 的參數，用來設定會話選項。預設為 `{}`。詳細設定方式請參考：[SessionOptions](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions)。
    - **provider_option** (`Dict[str, Any]`)：這是 onnxruntime.provider_options 的參數。預設為 `{}`。詳細設定方式請參考：[CUDAExecutionProvider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options)。

- **推論**

    在載入模型時，該函數會載入 ONNX 檔案內的資訊，並為輸入和輸出值給定一個字典，其中包含輸入和輸出的形狀和數據類型。

    因此，當你呼叫 `ONNXEngine` 實例時，你可以直接使用該字典來得到輸出結果。

- **範例**

    ```python
    import docsaidkit as D

    model_path = 'model.onnx'
    engine = D.ONNXEngine(model_path)
    print(engine)

    # Inferencing
    # Assuming the model has two inputs and two outputs and named:
    #   'input1', 'input2', 'output1', 'output2'.
    input_data = {
        'input1': np.random.randn(1, 3, 224, 224).astype(np.float32),
        'input2': np.random.randn(1, 3, 224, 224).astype(np.float32),
    }

    outputs = engine(**input_data)

    output_data1 = outputs['output1']
    output_data2 = outputs['output2']
    ```

