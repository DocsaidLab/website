---
sidebar_position: 3
---

# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, \*\*kwargs)](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/onnxengine/metadata.py#L20)

- **説明**：カスタムメタデータを ONNX モデルに書き込みます。

- **パラメータ**

  - **onnx_path** (`Union[str, Path]`)：ONNX モデルのパス。
  - **out_path** (`Union[str, Path]`)：出力される ONNX モデルのパス。
  - **drop_old_meta** (`bool`)：古いメタデータを削除するかどうか。デフォルトは `False`。
  - `**kwargs`：カスタムメタデータ。

- **例**

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
