---
slug: gosu-usage
title: コンテナ内のユーザー切り替えツール：gosu
authors: Zephyr
tags: [docker, gosu]
image: /ja/img/2024/0203.webp
description: 便利なツール gosu を学びましょう。
---

Docker 技術は、デプロイや管理において広く利用されています。

私たちは通常、さまざまなアプリケーションや関連する依存関係を一緒にパッケージ化し、異なる環境で一貫して動作させることを確保します。

<!-- truncate -->

## よくある問題

しかし、Docker を頻繁に使用していると、いくつかのよくある問題を避けるのは難しいです。

### TTY の変換

一般的な状況として、コンテナ内でファイルを出力し、その後コンテナから抜けると、出力したファイルの権限がすべて `root` になっていることがあります。

その場合、毎回 `chown` コマンドを使ってファイルの権限を変更する必要があり、とても煩わしい作業になります。

---

または、Docker コンテナ内で `sudo` を使ってターミナルとの対話を必要とするアプリケーションを起動すると、ターミナル（TTY）を正しく認識できないことがあります。これは、`sudo` が新しいセッションを作成する際に、ターミナルの所有権や制御を適切に処理しない可能性があるためです。

結果として、これらのアプリケーションが正常に動作しない、または入出力エラーが発生することがあります。

### シグナルの転送

例えば、Web サーバー（Apache や Nginx など）を実行するコンテナがあるとします。通常、コマンドラインツールを使ってコンテナを管理しますが、`sudo` を使って Web サーバーを起動すると、`sudo` が新しいプロセスを作成するため、停止や再起動のシグナルが Web サーバーに届かないことがあります。

:::tip
`sudo` の設計は主にセキュリティを向上させるためのもので、通常は一般ユーザーが `root` 権限でコマンドを実行できるようにするものです。しかし、Docker のような軽量仮想化環境では、`sudo` が作成する新しいセッションとシグナル伝達の方法が適合しない場合があります。
:::

## gosu とは？

- [**gosu GitHub リポジトリ**](https://github.com/tianon/gosu)

`gosu` はコンテナ内でのコマンド実行を簡単かつ安全にするために設計されたツールです。特に、`root` ユーザーから非特権ユーザーへ切り替えてプログラムを実行する必要がある場合に役立ちます。`libcontainer` のコードを基に設計されており、`/etc/passwd` の処理を直接活用しています。

簡単に言えば、`gosu` は「このユーザーでこのコマンドを実行して」と指示すると、それを実行し、処理後は何の痕跡も残さず終了する便利なツールです。

### 実際の利用シナリオ

`gosu` の最も典型的な利用例は、Docker コンテナの ENTRYPOINT スクリプト内で `root` ユーザーから非特権ユーザーに切り替えて操作を行う場合です。この方法は、コンテナのセキュリティを向上させるために非常に有効です。

`gosu` のインストールは非常に簡単で、Dockerfile に数行の指示を追加するだけで完了します。以下はその例です：

```Dockerfile title="Dockerfile"
# 既存のベースイメージを使用
FROM some_base_image:latest

WORKDIR /app

# gosu をインストール
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# ENTRYPOINT スクリプトを準備
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["default_command"]
```

以下は `entrypoint.sh` スクリプトの例です。このスクリプトでは、環境変数 `USER_ID` と `GROUP_ID` を基に動的にユーザーを作成し、そのユーザーでコマンドを実行します：

```bash title="entrypoint.sh"
#!/bin/bash
# 環境変数 USER_ID と GROUP_ID が設定されているか確認
if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then
    # ユーザーグループとユーザーを作成
    groupadd -g "$GROUP_ID" usergroup
    useradd -u "$USER_ID" -g usergroup -m user
    # gosu を使ってコマンドを実行
    exec gosu user "$@"
else
    exec "$@"
fi
```

詳細な例については以下を参考にしてください：[**Example training docker**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile)

### セキュリティに関する考慮事項

`gosu` は `root` ユーザーから非特権ユーザーに切り替えるための便利なツールですが、開発者は `gosu` の使用シナリオを十分に理解し、安全な環境でのみ使用するよう注意する必要があります。不適切な使用はセキュリティリスクを引き起こす可能性があります。
