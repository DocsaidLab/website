# JPG Process

> [get_orientation_code(stream: Union[str, Path, bytes]) -> Union[ROTATE, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L34)

> [jpgencode(img: np.ndarray, quality: int = 90) -> Union[bytes, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L50)

> [jpgdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L60)

> [jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L72)

この一連の関数は、JPG 画像のエンコード、デコード、読み取りをサポートし、EXIF データから自動的に画像の向きを調整する機能も提供します。

## 説明

- **get_orientation_code**：画像の EXIF データから向き情報を抽出し、画像の回転に適したコードに変換します。このステップは`jpgdecode`および`jpgread`関数内で自動的に行われ、読み込んだ画像が正しい向きで表示されることを保証します。
- **jpgencode**：NumPy 画像配列を JPG 形式のバイト列にエンコードします。`jpgencode`関数を使用する際は、`quality`パラメータを調整して画像の品質とファイルサイズのバランスを取ることができます。
- **jpgdecode**：JPG 形式のバイト列を NumPy 画像配列にデコードし、EXIF データに基づいて画像の向きを調整します。
- **jpgread**：JPG 画像ファイルを読み込み、NumPy 画像配列にデコードし、EXIF データに基づいて画像の向きを調整します。

## パラメータ

### jpgencode

- **img** (`np.ndarray`)：エンコードする画像配列。
- **quality** (`int`)：エンコード品質、範囲は 1 から 100。デフォルトは 90。

### jpgdecode

- **byte\_** (`bytes`)：デコードする JPG 形式のバイト列。

### jpgread

- **img_file** (`Union[str, Path]`)：読み込む JPG 画像ファイルのパス。

## 使用例

### jpgencode

```python
import numpy as np
import capybara as cb

img_array = np.random.rand(100, 100, 3) * 255
encoded_bytes = cb.jpgencode(img_array, quality=95)
```

### jpgdecode

```python
decoded_img = cb.jpgdecode(encoded_bytes)
```

### jpgread

```python
img_array = cb.jpgread('path/to/image.jpg')
```

## 追加の説明：TurboJPEG

[**TurboJPEG**](https://github.com/libjpeg-turbo/libjpeg-turbo)は、JPEG 画像処理ライブラリで、迅速な画像のエンコード、デコード、圧縮、解凍機能を提供します。`jpgencode`および`jpgdecode`関数では、JPEG 画像のエンコードとデコードに`TurboJPEG`を使用しています。`TurboJPEG`は`libjpeg-turbo`の Python ラッパーで、高速な画像処理を提供し、さまざまな画像フォーマットをサポートします。

- **特徴**

  - **高効率**：libjpeg-turbo ライブラリの高性能特性を活用し、従来の JPEG 処理方法に比べて画像処理速度が大幅に向上します。
  - **使いやすさ**：シンプルな API を提供し、開発者は効率的に JPEG 画像処理を実装できます。
  - **柔軟性**：さまざまな画像品質や圧縮レベルの設定に対応し、異なるシーンでの画像品質とファイルサイズのニーズに対応します。
  - **クロスプラットフォーム**：Windows、macOS、Linux などの複数のオペレーティングシステムをサポートし、異なる開発環境で使用できます。

インストール後、Python で TurboJPEG を使用してエンコードとデコードの機能を利用できます：

```python
from turbojpeg import TurboJPEG

# TurboJPEGインスタンスを初期化
jpeg = TurboJPEG()

# デコード
bgr_array = jpeg.decode(byte_)

# エンコード
byte_ = jpeg.encode(img, quality=quality)
```
