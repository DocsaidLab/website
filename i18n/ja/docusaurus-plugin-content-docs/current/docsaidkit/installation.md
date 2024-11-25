---
sidebar_position: 2
---

# 基本インストール

DocsaidKit のインストールを始める前に、以下のシステム要件を満たしていることを確認してください：

## 前提条件

### Python バージョン

- Python 3.8 以上がインストールされていることを確認してください。

### 依存パッケージ

使用しているオペレーティングシステムに応じて、必要な依存パッケージをインストールします。

- **Ubuntu**

  ターミナルを開き、以下のコマンドで依存パッケージをインストールします：

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  Brew を使って依存パッケージをインストールします：

  ```bash
  brew install jpeg-turbo exiftool ffmpeg libheif
  ```

### pdf2image 依存パッケージ

pdf2image は PDF ファイルを画像に変換するための Python モジュールです。

使用しているオペレーティングシステムに応じて、以下の手順でインストールします：

- または、オープンソースプロジェクト [**pdf2image**](https://github.com/Belval/pdf2image) の関連ページを参照してインストールガイドを確認してください。

- MacOS：Mac のユーザーは poppler をインストールする必要があります。Brew を使用してインストールできます：

  ```bash
  brew install poppler
  ```

- Linux：ほとんどの Linux ディストリビューションには `pdftoppm` と `pdftocairo` がプリインストールされています。

  インストールされていない場合は、パッケージマネージャーを使用して poppler-utils をインストールします：

  ```bash
  sudo apt install poppler-utils
  ```

## パッケージのインストール

前提条件が整ったら、次に git clone を使用してインストールします：

1. このパッケージをダウンロード：

   ```bash
   git clone https://github.com/DocsaidLab/DocsaidKit.git
   ```

2. wheel パッケージをインストール：

   ```bash
   pip install wheel
   ```

3. wheel ファイルをビルド：

   ```bash
   cd DocsaidKit
   python setup.py bdist_wheel
   ```

4. ビルドした wheel パッケージをインストール：

   ```bash
   pip install dist/docsaidkit-*-py3-none-any.whl
   ```

   PyTorch 対応のバージョンをインストールする場合は、次のコマンドを使用します：

   ```bash
   pip install "dist/docsaidKit-${version}-none-any.whl[torch]"
   ```

## よくある質問

1. **なぜ Windows には対応していないのですか？**

   安全のため、Windows は避けるべきです。

2. **Windows を使いたいです。余計なことを言わないでください！**

   わかりました。Windows ユーザーには、Docker をインストールし、上記の方法で Docker を使用してプログラムを実行することをお勧めします。

   次の章：[**進階インストール**](./advance.md)を参照してください。

3. **Docker のインストール方法は？**

   難しくはありませんが、手順が少し多いです。

   [**Docker 公式ドキュメント**](https://docs.docker.com/get-docker/) を参照してインストールを行ってください。
