---
slug: convert-pdf-to-images
title: Pythonを使用してPDFを画像に変換する
authors: Z. Yuan
tags: [Python, pdf2image]
image: /ja/img/2024/0214.webp
description: オープンソースの`pdf2image`パッケージを使用して問題を解決する方法。
---

開発中に、PDFファイルを画像フォーマットに変換する必要がよくあります。これは、ドキュメントの表示、データ処理、またはコンテンツの共有に使用されます。

この記事では、PDFファイルをPIL画像に変換できる便利なPythonモジュール、[**pdf2image**](https://github.com/Belval/pdf2image/tree/master) を紹介します。

<!-- truncate -->

## 依存関係のインストール

`pdf2image`は、`pdftoppm`と`pdftocairo`という2つのツールに依存しています。異なるオペレーティングシステムにおけるインストール方法は若干異なります：

- **Mac**：Homebrewを使用してPopplerをインストールするには、以下のコマンドを実行します：

  ```shell
  brew install poppler
  ```

- **Linux**：ほとんどのLinuxディストリビューションには、`pdftoppm`と`pdftocairo`が事前にインストールされています。もしインストールされていない場合は、以下のコマンドでインストールできます：

  ```shell
  sudo apt-get install poppler-utils   # Ubuntu/Debianシステム
  ```

- **`conda`を使用する**：どのプラットフォームでも、`conda`を使ってPopplerをインストールできます：

  ```shell
  conda install -c conda-forge poppler
  ```

インストールが完了したら、`pdf2image`をインストールします。

## `pdf2image`のインストール

以下のコマンドを実行して、`pdf2image`をインストールします：

```shell
pip install pdf2image
```

## 使用方法

PDFを画像に変換する基本的な使い方は非常にシンプルです。

以下の例では、PDFの各ページをPIL画像オブジェクトに変換し、画像ファイルとして保存する方法を示します：

```python
from pdf2image import convert_from_path

# PDFファイルを画像のリストに変換
images = convert_from_path('/path/to/your/pdf/file.pdf')

# 各ページをPNG形式で保存
for i, image in enumerate(images):
    image.save(f'output_page_{i+1}.png', 'PNG')
```

もしバイナリデータから変換したい場合は、以下のようにします：

```python
with open('/path/to/your/pdf/file.pdf', 'rb') as f:
    pdf_data = f.read()

images = convert_from_bytes(pdf_data)
```

## オプションと高度な設定

`pdf2image`は、画像の品質や範囲をカスタマイズできる多くのオプションを提供しています：

- **DPI設定**：`dpi`パラメータを調整すると、画像の解像度を向上させることができ、高品質な画像が必要な場合に便利です：

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', dpi=300)
  ```

- **ページ範囲の指定**：`first_page`および`last_page`パラメータを使用すると、特定のページだけを変換できます：

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', first_page=2, last_page=5)
  ```

- **出力画像フォーマット**：`fmt`パラメータを使って、出力する画像のフォーマットをJPEGやPNGなどに指定できます：

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', fmt='jpeg')
  ```

- **エラーハンドリング**：変換中にフォーマットエラーやファイル破損が発生することがあります。`try/except`を使用して例外をキャッチすることをお勧めします：

  ```python
  try:
      images = convert_from_path('/path/to/your/pdf/file.pdf')
  except Exception as e:
      print("変換失敗：", e)
  ```

## 結論

`pdf2image`は非常に便利なツールで、より多くのオプションや詳細な使い方については[**pdf2image公式ドキュメント**](https://github.com/Belval/pdf2image/tree/master)を参照してください。