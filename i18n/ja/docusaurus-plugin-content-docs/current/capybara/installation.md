---
sidebar_position: 2
---

# 基本インストール

Capybara のインストールを開始する前に、システムが以下の要件を満たしていることを確認してください。

## 前提条件

インストール前に、システムに Python 3.10 以上がインストールされていることを確認してください。

開発は Ubuntu オペレーティングシステムを基に行っていますので、以下のガイドラインは Windows および MacOS ユーザーには適用されない場合があります。

次に、ターミナルを開き、以下のコマンドを実行して依存関係をインストールします：

```bash
sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
```

### pdf2image 依存パッケージ

pdf2image は PDF ファイルを画像に変換するための Python モジュールです。

使用しているオペレーティングシステムに応じて、以下の指示に従ってインストールしてください：

- また、オープンソースプロジェクト [**pdf2image**](https://github.com/Belval/pdf2image) のページを参照して、インストールガイドを取得できます。

多くの Linux ディストリビューションには `pdftoppm` と `pdftocairo` が事前にインストールされています。

もしインストールされていない場合は、パッケージマネージャーを使って poppler-utils をインストールしてください。

```bash
sudo apt install poppler-utils
```

## パッケージのインストール

前提条件を満たしたら、git clone を使ってインストールできます：

1. このパッケージをダウンロード：

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. wheel パッケージをインストール：

   ```bash
   pip install wheel
   ```

3. wheel ファイルをビルド：

   ```bash
   cd Capybara
   python setup.py bdist_wheel
   ```

4. ビルドした wheel パッケージをインストール：

   ```bash
   pip install dist/capybara-*-py3-none-any.whl
   ```

## よくある質問

1. **Windows はサポートされていますか？**

   命を大切にしましょう、Windows は避けてください。

2. **どうしても Windows を使いたいんです、余計なこと言わないでください！**

   わかりました、その場合 Docker をインストールし、上記の方法で Docker 経由でプログラムを実行することをお勧めします。

   次の記事を参照してください：[**進階インストール**](./advance.md)。

3. **Docker はどうやってインストールしますか？**

   難しくはありませんが、手順が少し多いです。

   インストール方法については [**Docker の公式ドキュメント**](https://docs.docker.com/get-docker/) を参照してください。
