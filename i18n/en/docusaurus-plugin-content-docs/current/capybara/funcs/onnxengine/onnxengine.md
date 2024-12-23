# ONNXEngine

> [ONNXEngine(model_path: Union[str, Path], gpu_id: int = 0, backend: Union[str, int, Backend] = Backend.cpu, session_option: Dict[str, Any] = {}, provider_option: Dict[str, Any] = {})](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/onnxengine/engine.py#L18)

- **Description**: Initializes the ONNX model inference engine.

- **Parameters**

  - **model_path** (`Union[str, Path]`): The file name or serialized ONNX or ORT format model byte string.
  - **gpu_id** (`int`): The GPU ID. Default is 0.
  - **backend** (`Union[str, int, Backend]`): The inference backend, which can be `Backend.cpu` or `Backend.cuda`. Default is `Backend.cpu`.
  - **session_option** (`Dict[str, Any]`): Parameters for `onnxruntime.SessionOptions` to set session options. Default is `{}`. For detailed configuration, refer to: [SessionOptions](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions).
  - **provider_option** (`Dict[str, Any]`): Parameters for `onnxruntime.provider_options`. Default is `{}`. For detailed configuration, refer to: [CUDAExecutionProvider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options).

- **Inference**

  When loading the model, this function loads the information inside the ONNX file and provides a dictionary with input and output shapes and data types.

  Therefore, when calling the `ONNXEngine` instance, you can directly use this dictionary to obtain the output results.

- **Example**

  ```python
  import capybara as cb

  model_path = 'model.onnx'
  engine = cb.ONNXEngine(model_path)
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
