---
sidebar_position: 4
---

# JPG Process

> [get_orientation_code(stream: Union[str, Path, bytes]) -> Union[ROTATE, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L34C5-L34C25)

> [jpgencode(img: np.ndarray, quality: int = 90) -> Union[bytes, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L50)

> [jpgdecode(byte\_: bytes) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L60)

> [jpgread(img_file: Union[str, Path]) -> Union[np.ndarray, None]](https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/vision/improc.py#L72)

この一連の関数は、JPG 画像のエンコード、デコード、読み込みをサポートし、EXIF データから画像の向きを自動的に調整する機能も提供します。

## 説明

- **get_orientation_code**：画像の EXIF データから方向情報を抽出し、画像の回転に適したコードに変換します。このステップは、`jpgdecode`および`jpgread`関数内で自動的に行われ、読み込まれた画像が正しい向きで表示されるようにします。
- **jpgencode**：NumPy 画像配列を JPG 形式のバイト列にエンコードします。`jpgencode`関数を使用する際は、`quality`パラメータを調整することで、画像品質とファイルサイズのバランスを取ることができます。
- **jpgdecode**：JPG 形式のバイト列を NumPy 画像配列にデコードし、EXIF データに基づいて画像の向きを調整します。
- **jpgread**：ファイルから JPG 画像を読み込み、NumPy 画像配列としてデコードし、EXIF データに基づいて画像の向きを調整します。

## パラメータ

### jpgencode

- **img** (`np.ndarray`)：エンコードする画像配列。
- **quality** (`int`)：エンコード品質。1 から 100 の範囲。デフォルトは 90。

### jpgdecode

- **byte\_** (`bytes`)：デコードする JPG 形式のバイト列。

### jpgread

- **img_file** (`Union[str, Path]`)：読み込む JPG 画像ファイルのパス。

## 例

### jpgencode

```python
import numpy as np
import docsaidkit as D

img_array = np.random.rand(100, 100, 3) * 255
encoded_bytes = D.jpgencode(img_array, quality=95)
```

### jpgdecode

```python
decoded_img = D.jpgdecode(encoded_bytes)
```

### jpgread

```python
img_array = D.jpgread('path/to/image.jpg')
```

## 追加の説明：TurboJPEG

[**TurboJPEG**](https://github.com/libjpeg-turbo/libjpeg-turbo)は、高速な JPEG 画像処理ライブラリで、画像のエンコード、デコード、圧縮および解凍機能を提供します。`jpgencode`および`jpgdecode`関数では、JPEG 画像のエンコードとデコードに`TurboJPEG`を使用しています。`TurboJPEG`は、`libjpeg-turbo`の Python ラッパーであり、より高速な画像エンコードとデコードを提供し、さまざまな画像形式をサポートします。

- **特徴**
  - **高効率**：`libjpeg-turbo`ライブラリの高性能機能を活用し、従来の JPEG 処理方法に比べて大幅に高速化されています。
  - **使いやすさ**：簡潔で明確な API を提供しており、開発者が簡単に効率的な JPEG 画像処理を実装できます。
  - **柔軟性**：さまざまな画像品質や圧縮レベルの設定をサポートし、異なるシナリオでの画像品質とファイルサイズのニーズに対応します。
  - **クロスプラットフォーム**：Windows、macOS、Linux など、複数の OS でサポートされており、異なる開発環境で使用できます。

インストール後、Python で`TurboJPEG`を使用してエンコードおよびデコード機能を利用するには、以下のようにします：

```python
from turbojpeg import TurboJPEG

# TurboJPEGインスタンスを初期化
jpeg = TurboJPEG()

# デコード
bgr_array = jpeg.decode(byte_)

# エンコード
byte_ = jpeg.encode(img, quality=quality)
```
