---
sidebar_position: 2
---

# get_onnx_metadata

> [get_onnx_metadata(onnx_path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/onnxengine/metadata.py#L10)

- **Description**

    Retrieve custom metadata from an ONNX model.

- **Parameters**
    - **onnx_path** (`Union[str, Path]`): Path to the ONNX model.

- **Returns**
    - **dict**: Custom metadata.

- **Example**

    ```python
    import docsaidkit as D

    onnx_path = 'model.onnx'
    metadata = D.get_onnx_metadata(onnx_path)
    print(metadata)
    # >>> metadata = {
    #     'key1': 'value1',
    #     'key2': 'value2',
    #     'key3': 'value3',
    # }
    ```
