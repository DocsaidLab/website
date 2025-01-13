---
slug: convert-pdf-to-images
title: Python で PDF を画像に変換する
authors: Zephyr
tags: [Python, pdf2image]
image: /ja/img/2024/0214.webp
description: オープンソースライブラリ pdf2image を使用して問題を解決します。
---

PDF ファイルを画像形式に変換する必要があることはよくあります。

ここでは、便利な Python モジュール [**pdf2image**](https://github.com/Belval/pdf2image/tree/master) をお勧めします。このモジュールを使用すると、PDF ファイルを PIL 画像に変換できます。

<!-- truncate -->

## 依存関係のインストール

`pdf2image` は `pdftoppm` および `pdftocairo` に依存しています。使用しているオペレーティングシステムによってインストール方法が異なります：

- **Mac**：Homebrew を使用して Poppler をインストール：`brew install poppler`。
- **Linux**：多くの Linux ディストリビューションでは、`pdftoppm` と `pdftocairo` がすでにインストールされています。インストールされていない場合は、パッケージマネージャーを使用して `poppler-utils` をインストールしてください。
- **`conda` を使用する場合**：どのプラットフォームでも、`conda` を使用して Poppler をインストールできます：`conda install -c conda-forge poppler`。その後、`pdf2image` をインストールしてください。

## `pdf2image` のインストール

まず、`pdf2image` をインストールします。以下のコマンドをターミナルで実行してください：

```shell
pip install pdf2image
```

## `pdf2image` を使って PDF を変換する

PDF を画像に変換する基本的な方法は非常に簡単です：

```python
from pdf2image import convert_from_path

images = convert_from_path('/path/to/your/pdf/file.pdf')
```

このコードは、PDF の各ページを PIL 画像オブジェクトに変換し、それを `images` リストに保存します。

バイナリデータから PDF を変換することもできます：

```python
images = convert_from_bytes(open('/path/to/your/pdf/file.pdf', 'rb').read())
```

## オプションパラメータ

`pdf2image` は豊富なオプションパラメータを提供しており、DPI、出力形式、ページ範囲などをカスタマイズできます。例えば、`dpi=300` を使用して出力画像の解像度を向上させたり、`first_page` と `last_page` を使用して変換範囲を指定できます。

詳細については以下を参照してください：

- `pdf2image` の[**公式ドキュメント**](https://github.com/Belval/pdf2image/tree/master)；

## 結論

`pdf2image` は非常に強力で使いやすいツールであり、PDF を画像に変換する必要がある場合に最適なソリューションを提供します。文書処理、データ整理、コンテンツ表示など、さまざまな用途で効率的に利用できます。
