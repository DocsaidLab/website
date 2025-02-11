---
slug: flexible-video-conversion-by-python
title: バッチ動画変換
authors: Z. Yuan
image: /ja/img/2024/1217.webp
tags: [Media-Processing, Python, ffmpeg]
description: Python と ffmpeg を使用して指定された形式のバッチ変換プロセスを作成する。
---

MOV 形式の動画ファイルを一括で受け取ったが、システムでは読み取れないため、MP4 に変換する必要がある。

仕方なく、自分でプログラムを書くことにした。

<!-- truncate -->

## 設計草案

変換ツールとして間違いなく選ばれるのは ffmpeg です。このオープンソースツールはほぼすべてのメディア形式をサポートしており、コマンドライン引数で変換方法を制御できます。

最初はこの機能をフロントエンドに直接組み込んで、他の人が自由に変換できるようにしようと思いましたが...

ブラウザでの呼び出しに問題が発生し、1 時間試しても解決できなかったため、結局ローカルで処理することにしました。

ローカルでの変換は比較的簡単で、最初は Bash で書こうと思いましたが、Python の方がメンテナンスしやすいと感じたので、最終的に Python と ffmpeg を使ってこの機能を実装しました。

## FFMPEG とは？

[ffmpeg](https://ffmpeg.org/) は非常に強力なオープンソースのマルチメディア処理ツールで、広く音声・映像形式の変換、ストリーミング、編集、結合など多くの作業に利用されています。

多くの一般的および珍しいメディア形式をサポートしており、内蔵された多くのコーデックを使用して、コマンドライン操作で素早く変換、カット、字幕埋め込み、再サンプリング、圧縮、およびクロスプラットフォームのメディアストリーミングを行うことができます。

ffmpeg はオープンソースプロジェクトであり、Linux、macOS、Windows などさまざまな OS に簡単にインストールして実行できるため、メディア関連のワークフローで欠かせないツールとなっています。

一般的なシナリオでは、簡単なコマンドで最も一般的な変換を実行できます。たとえば、MOV ファイルを MP4 に変換する場合は、次のようにコマンドを実行します：

```bash
ffmpeg -i input.mov -c copy output.mp4
```

ここで、`-i`は入力ファイルのパスを指定し、`-c copy`はソースファイルの音声・映像トラックをそのままコピーすることを意味します（再エンコードせず）。これにより、処理時間が大幅に短縮され、元の品質を維持できます。品質、エンコード設定、出力解像度、ビットレート、チャンネル数などを調整したい場合、ffmpeg はかなり柔軟なコマンドライン引数を提供し、カスタマイズできます。

要するに、非常に優れたツールで、ぜひ覚えておくべきです！

## 環境準備

今回は Ubuntu OS をベースに開発を行いますが、似たような Linux システムでも使用可能です。

1. **Python 環境**：Python 3.x がインストールされていることを確認します：

   ```bash
   python3 --version
   ```

2. **ffmpeg のインストール**：Ubuntu 環境では次のコマンドでインストールできます：

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   インストール後、バージョンを確認します：

   ```bash
   ffmpeg -version
   ```

3. **プログラム構成**：プロジェクトフォルダ内に`convert.py`（ファイル名は任意）を作成し、下記のコードを貼り付けます。

## コード例

```python
import subprocess
import sys
from pathlib import Path

def convert_videos(input_dir: Path, src_format: str, dest_format: str):
    # 目標フォルダが存在するか確認
    if not input_dir.is_dir():
        print(f"エラー: フォルダ '{input_dir}' は存在しません。")
        sys.exit(1)

    # 自動的に出力フォルダを作成
    output_dir = input_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # フォーマットのプレフィックスを追加
    if not src_format.startswith("."):
        src_format = f".{src_format}"
    if not dest_format.startswith("."):
        dest_format = f".{dest_format}"

    # 指定された形式のファイルを探索（大文字小文字を区別しない）
    video_files = [f for f in input_dir.rglob("*") if f.suffix.casefold() == src_format.casefold()]

    if not video_files:
        print(f"{src_format}ファイルは見つかりませんでした。")
        sys.exit(0)

    for file in video_files:
        output_file = output_dir / f"{file.stem}{dest_format}"
        print(f"変換中: '{file}' -> '{output_file}'")

        # ffmpegを使ってファイルを変換
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(file),
                    "-c", "copy",
                    str(output_file)
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"変換成功: '{output_file}'")
        except subprocess.CalledProcessError as e:
            print(f"変換失敗: '{file}'")
            print(e.stderr.decode())

    print(f"すべてのファイルが処理されました。出力フォルダ: '{output_dir}'")


if __name__ == "__main__":
    # 引数が正しいか確認
    if len(sys.argv) != 4:
        print(f"使用法: python {sys.argv[0]} <目標フォルダ> <元の形式> <目標形式>")
        print(f"例: python {sys.argv[0]} 'videos' 'MOV' 'mp4'")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    src_format = sys.argv[2]
    dest_format = sys.argv[3]

    convert_videos(input_dir, src_format, dest_format)
```

## 使用方法

1. **ソースファイルを準備**：変換したいファイル（MOV、AVI、MKV など）を指定されたフォルダ（例：`videos`）に入れます。

2. **変換を実行**：プログラムファイルがあるディレクトリに移動し、次のコマンドを実行します：

   ```bash
   python3 convert.py videos MOV mp4
   ```

   例えば、AVI ファイルを MKV に変換したい場合は次のようにします：

   ```bash
   python3 convert.py videos avi mkv
   ```

   実行後、プログラムは`videos/output`フォルダ内に変換されたファイルを作成します。

3. **結果を確認**：`output`フォルダ内に正しい形式で変換された動画が存在することを確認し、作業が完了したことを確認します。

## 高度な使い方

ファイルを圧縮して品質を調整したい場合は、ffmpeg コマンドに特定のオプションを追加できます。例えば：

```bash
ffmpeg -i input.avi -c:v libx264 -crf 20 output.mp4
```

そして、プログラム内で ffmpeg の呼び出し方法を調整します。

## 結論

以上です。開発中にサッと書いたシンプルな機能ですが、あなたにも役立つことを願っています。

さあ、変換を開始しましょう！
