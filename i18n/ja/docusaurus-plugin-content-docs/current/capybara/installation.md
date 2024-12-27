---
sidebar_position: 2
---

# 基本インストール

Capybara をインストールする前に、システムが以下の要件を満たしていることを確認してください：

## 依存パッケージ

オペレーティングシステムに応じて、以下の必須システムパッケージをインストールしてください：

- **Ubuntu**

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  ```bash
  brew install jpeg-turbo exiftool ffmpeg
  ```

  - **特記事項**：テスト結果として、macOS で libheif を使用する際にいくつかの既知の問題があります。主な問題は以下の通りです：

    1. **生成された HEIC ファイルが開けない**：macOS 上で libheif で生成された HEIC ファイルが一部のアプリケーションで開けないことがあります。これは画像サイズに関連しており、特に画像の幅や高さが奇数の場合に互換性の問題が発生することがあります。

    2. **コンパイルエラー**：macOS で libheif をコンパイルする際に、ffmpeg デコーダーに関連する未定義シンボルエラーが発生することがあります。これはコンパイルオプションや依存関係の設定ミスによるものです。

    3. **サンプルプログラムが実行できない**：macOS Sonoma では、libheif のサンプルプログラムが正常に動作せず、`libheif.1.dylib`が見つからないというダイナミックリンクエラーが発生することがあります。これは動的ライブラリのパス設定に関する問題です。

    これらの問題が多いため、現在は Ubuntu でのみ libheif を使用しています。macOS に関しては、将来のバージョンで解決される予定です。

### pdf2image

pdf2image は PDF ファイルを画像に変換する Python モジュールで、システムに以下のツールがインストールされていることを確認してください：

- MacOS：poppler をインストールする必要があります

  ```bash
  brew install poppler
  ```

- Linux：ほとんどのディストリビューションには`pdftoppm`と`pdftocairo`がデフォルトでインストールされています。もしインストールされていない場合、以下を実行してください：

  ```bash
  sudo apt install poppler-utils
  ```

### ONNXRuntime

ONNXRuntime を使用して GPU 加速推論を行う場合、適切な CUDA バージョンがインストールされていることを確認してください。以下の手順でインストールできます：

```bash
sudo apt install cuda-12-4
# .bashrcに追加する場合
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```

## PyPI を通じてインストール

1. PyPI を使ってパッケージをインストール：

   ```bash
   pip install capybara-docsaid
   ```

2. インストール確認：

   ```bash
   python -c "import capybara; print(capybara.__version__)"
   ```

3. バージョン番号が表示された場合、インストールは成功です。

## git clone を通じてインストール

1. プロジェクトをダウンロード：

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

4. ビルドされた wheel ファイルをインストール：

   ```bash
   pip install dist/capybara_docsaid-*-py3-none-any.whl
   ```

## よくある質問

1. **なぜ Windows のインストールはサポートされていないのですか？**

   命を大事に、Windows からは離れましょう。

---

2. **Windows を使いたいんですが、余計なことは言わないで！**

   わかりました、Docker をインストールし、上記の手順で Docker を使ってプログラムを実行することをお勧めします。

   詳しくは次のページを参照：[**詳細インストール**](./advance.md)。

---

3. **Docker はどうやってインストールするのですか？**

   難しくはありませんが、手順がいくつかあります。

   インストール方法については[**Docker 公式ドキュメント**](https://docs.docker.com/get-docker/)を参照してください。
