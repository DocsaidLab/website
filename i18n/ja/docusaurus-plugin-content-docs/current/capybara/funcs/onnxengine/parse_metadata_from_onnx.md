# parse_metadata_from_onnx

> [parse_metadata_from_onnx(onnx_path: str | Path) -> dict[str, Any]](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **説明**：ONNX の `custom_metadata_map` を読み取り、文字列フィールドに対して `json.loads()` を試みて元の型へ戻します。

- **依存関係**

  - `onnxruntime` が必要です（`InferenceSession` 経由で metadata を読み取ります）。

- **パラメータ**

  - **onnx_path** (`str | Path`)：ONNX モデルのパス。

- **戻り値**

  - **dict[str, Any]**：パース済み metadata。

- **例**

  ```python
  from capybara.onnxengine import parse_metadata_from_onnx

  meta = parse_metadata_from_onnx("model.onnx")
  print(meta)
  ```

