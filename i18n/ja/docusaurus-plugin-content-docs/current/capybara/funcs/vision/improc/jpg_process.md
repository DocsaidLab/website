# JPG Process

> [get_orientation_code(stream: str | Path | bytes) -> ROTATE | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgencode(img: np.ndarray, quality: int = 90) -> bytes | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgdecode(byte\_: bytes) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

> [jpgread(img_file: str | Path) -> np.ndarray | None](https://github.com/DocsaidLab/Capybara/blob/main/capybara/vision/improc.py)

この一連の関数は、JPG 画像のエンコード／デコード／読み取り、および EXIF からの自動回転（向き補正）を提供します。

## 説明

- **get_orientation_code**：画像の EXIF から向き情報を取り出し、回転コードに変換します（`jpgdecode` と `jpgread` 内で自動的に使われます）。
- **jpgencode**：NumPy 画像配列を JPG bytes にエンコードします。失敗時は `None`。
- **jpgdecode**：JPG bytes を NumPy 画像配列にデコードし、EXIF に基づいて向きを補正します。失敗時は `None`。
- **jpgread**：JPG ファイルを読み取り、NumPy 画像配列にデコードし、EXIF に基づいて向きを補正します。失敗時は `None`。

## 依存関係

- このモジュールは JPEG の編解碼に `turbojpeg`（PyTurboJPEG）を使用します。

## パラメータ

### jpgencode

- **img** (`np.ndarray`)：エンコードする画像配列。
- **quality** (`int`)：エンコード品質（1〜100）。デフォルトは 90。

### jpgdecode

- **byte\_** (`bytes`)：デコードする JPG bytes。

### jpgread

- **img_file** (`str | Path`)：読み込む JPG ファイルのパス。

## 例

### jpgencode

```python
from capybara.vision.improc import imread, jpgencode

img = imread('lena.jpg')
encoded_bytes = jpgencode(img, quality=95)
```

### jpgdecode

```python
from capybara.vision.improc import jpgdecode

decoded_img = jpgdecode(encoded_bytes)
```

### jpgread

```python
from capybara.vision.improc import jpgread

img_array = jpgread('path/to/image.jpg')
```

