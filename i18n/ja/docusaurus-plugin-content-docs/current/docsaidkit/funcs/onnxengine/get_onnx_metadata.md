---
sidebar_position: 2
---

# get_onnx_metadata

> [get_onnx_metadata(onnx_path: Union[str, Path]) -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/onnxengine/metadata.py#L10)

- **説明**：ONNX モデルからカスタムメタデータを取得します。

- **パラメータ**

  - **onnx_path** (`Union[str, Path]`)：ONNX モデルのパス。

- **戻り値**

  - **dict**：カスタムメタデータ。

- **例**

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
