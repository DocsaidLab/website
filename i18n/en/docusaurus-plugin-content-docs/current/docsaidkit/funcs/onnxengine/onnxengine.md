---
sidebar_position: 1
---

# ONNXEngine

>[ONNXEngine(model_path: Union[str, Path], gpu_id: int = 0, backend: Union[str, int, Backend] = Backend.cpu, session_option: Dict[str, Any] = {}, provider_option: Dict[str, Any] = {})](https://github.com/DocsaidLab/DocsaidKit/blob/main/docsaidkit/onnxengine/engine.py)

- **Description**

    Initialize the ONNX model inference engine.

- **Parameters**

    - **model_path** (`Union[str, Path]`): The file name or byte string of the serialized ONNX or ORT format model.
    - **gpu_id** (`int`): GPU ID. Default is 0.
    - **backend** (`Union[str, int, Backend]`): Inference backend, can be `Backend.cpu` or `Backend.cuda`. Default is `Backend.cpu`.
    - **session_option** (`Dict[str, Any]`): Parameters for onnxruntime.SessionOptions to set session options. Default is `{}`. For detailed configuration, please refer to: [SessionOptions](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions).
    - **provider_option** (`Dict[str, Any]`): Parameters for onnxruntime.provider_options. Default is `{}`. For detailed configuration, please refer to: [CUDAExecutionProvider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options).

- **Inference**

    When loading the model, this function loads information from the ONNX file and gives a dictionary for input and output values, which includes shapes and data types for input and output.

    Therefore, when you call an `ONNXEngine` instance, you can directly use this dictionary to obtain the output results.

- **Example**

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
