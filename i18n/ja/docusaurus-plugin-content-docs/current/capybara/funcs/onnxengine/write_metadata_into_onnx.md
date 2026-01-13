# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: str | Path, out_path: str | Path, drop_old_meta: bool = False, **kwargs: Any) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **依存関係**

  - `onnx` が必要です（モデルファイルの read/write）。
  - `drop_old_meta=False` の場合、旧 metadata を読み取るために `onnxruntime` も必要です。

- **説明**：custom metadata を ONNX モデルへ書き込みます。

- **パラメータ**

  - **onnx_path** (`str | Path`)：ONNX モデルのパス。
  - **out_path** (`str | Path`)：出力 ONNX モデルのパス。
  - **drop_old_meta** (`bool`)：旧 metadata を削除するかどうか。デフォルトは `False`。
  - `**kwargs`：書き込む custom metadata。

- **挙動**

  - `Date` フィールドを自動追加します（`capybara.utils.time.now(fmt=...)` を使用）。
  - 各 metadata value は `json.dumps()` で文字列化して ONNX props に書き込みます。

- **例**

  ```python
  from capybara.onnxengine import write_metadata_into_onnx

  onnx_path = 'model.onnx'
  out_path = 'model_with_metadata.onnx'
  write_metadata_into_onnx(
      onnx_path,
      out_path,
      drop_old_meta=False,
      key1='value1',
      key2='value2',
      key3='value3',
  )
  ```

