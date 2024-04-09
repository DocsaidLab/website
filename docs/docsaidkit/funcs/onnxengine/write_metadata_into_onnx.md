---
sidebar_position: 3
---

# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, **kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/onnxengine/metadata.py#L20)

- **說明**：將自定義元數據寫入 ONNX 模型中。

- **參數**
    - **onnx_path** (`Union[str, Path]`)：ONNX 模型的路徑。
    - **out_path** (`Union[str, Path]`)：輸出 ONNX 模型的路徑。
    - **drop_old_meta** (`bool`)：是否刪除舊的元數據。預設為 `False`。
    - `**kwargs`：自定義元數據。

- **範例**

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

