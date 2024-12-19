---
sidebar_position: 3
---

# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, **kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/onnxengine/metadata.py#L20)

- **Description**

    Write custom metadata into an ONNX model.

- **Parameters**

    - **onnx_path** (`Union[str, Path]`): Path to the ONNX model.
    - **out_path** (`Union[str, Path]`): Path to the output ONNX model.
    - **drop_old_meta** (`bool`): Whether to drop old metadata. Default is `False`.
    - `**kwargs`: Custom metadata.

- **Example**

    ```python
    import docsaidkit as D

    onnx_path = 'model.onnx'
    out_path = 'model_with_metadata.onnx'
    D.write_metadata_into_onnx(
        onnx_path,
        out_path,
        drop_old_meta=False,
        key1='value1',
        key2='value2',
        key3='value3',
    )
    ```
