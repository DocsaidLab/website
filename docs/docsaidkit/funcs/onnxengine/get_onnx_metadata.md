---
sidebar_position: 2
---

# get_onnx_metadata

> [get_onnx_metadata(onnx_path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/onnxengine/metadata.py#L10)

- **說明**：從 ONNX 模型中取得自定義元數據。

- **參數**
    - **onnx_path** (`Union[str, Path]`)：ONNX 模型的路徑。

- **傳回值**

    - **dict**：自定義元數據。

- **範例**

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
