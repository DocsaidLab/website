# write_metadata_into_onnx

> [write_metadata_into_onnx(onnx_path: str | Path, out_path: str | Path, drop_old_meta: bool = False, **kwargs: Any) -> None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/onnxengine/metadata.py)

- **依賴**

  - 需要 `onnx`（讀寫模型檔）。
  - 當 `drop_old_meta=False` 時，會讀取舊 metadata，額外需要 `onnxruntime`。

- **說明**：將自定義元數據寫入 ONNX 模型中。

- **參數**

  - **onnx_path** (`Union[str, Path]`)：ONNX 模型的路徑。
  - **out_path** (`Union[str, Path]`)：輸出 ONNX 模型的路徑。
  - **drop_old_meta** (`bool`)：是否刪除舊的元數據。預設為 `False`。
  - `**kwargs`：自定義元數據。

- **行為**

  - 會自動加入 `Date` 欄位（使用 `capybara.utils.time.now(fmt=...)`）。
  - 會把每個 metadata value 以 `json.dumps()` 轉成字串寫入 ONNX props。

- **範例**

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
