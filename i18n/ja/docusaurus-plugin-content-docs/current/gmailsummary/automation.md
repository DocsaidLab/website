---
sidebar_position: 9
---

# スケジュールタスク

毎朝最新のメール要約を見ることができるように、完全に自動化されたスケジュールタスクを設定する必要があります。

## `crontab` の使用

このプロセスを完全に自動化するために、Linux の`crontab`機能を使用して定期的なタスクを設定します。

これにより、毎日決まった時間に自動的にプログラムが実行され、新しいメールを取得し、要約を生成して、GitHub リポジトリを更新することができます。

具体的な`crontab`の設定は以下の通りです：

```bash
crontab -e
```

次に、以下の内容を追加します：

```bash
# このファイルを編集して、cronで実行するタスクを追加します。
#
# 実行する各タスクは、単一行で定義する必要があります。
# どのフィールドでタスクが実行されるか、実行するコマンドを指定します。
#
# 時間を定義するために、分 (m)、時 (h)、日 (dom)、月 (mon)、曜日 (dow) の具体的な値を提供できます。
# それらのフィールドに '*' を使用することもできます（「任意」を意味します）。
#
# 注意：タスクはcronのシステムデーモンの時間とタイムゾーンに基づいて開始されます。
#
# crontabジョブの出力（エラーを含む）は、cronファイルが属するユーザーにメールで送信されます（リダイレクトされていない場合）。
#
# 例えば、毎週午前5時にすべてのユーザーアカウントをバックアップする場合：
# 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
#
# 詳細については、crontab(5) および cron(8) のマニュアルページを参照してください。
#
# m h  dom mon dow   command

# 環境変数を定義
OPENAI_API_KEY="your_openai_api_key"

# 毎日午前6時に自動的に更新スクリプトを実行
0 6 * * * /path/to/your/script/update_targets_infos.sh
```

定期タスクを設定する前に、スクリプトファイルに実行権限を与えることを忘れないでください：

```bash
chmod +x /path/to/your/script/update_targets_infos.sh
```

また、`crontab`の環境は特殊なため、実行する Python 環境と関連するライブラリが正しく設定されていることを確認する必要があります。

そのため、プログラム内では通常、Python プログラムを実行するために絶対パスを使用します。プログラム内のパスを修正することを忘れないでください。

```bash
# `update_targets_infos.sh`

# ...前略

# Pythonプログラムを実行するため、ここで自分のpythonのパスを指定します
$HOME/your/python main.py --project_name $project_name --time_length 1 2>&1

# ...後略
```

:::tip
`crontab`は`.bashrc`や`.bash_profile`などのファイルを読みませんので、プログラム内で環境変数をすべて指定する必要があります。

そのため、`crontab`で実行するプログラム内で`OPENAI_API_KEY`環境変数を設定しています。
:::

## `crontab`のテスト

設定が完了したら、`crontab`環境で自動化タスクが正常に動作するかどうかをテストする方法を考えましょう。

実行する方法の一つは、新しいターミナルを起動し、すべての環境変数を除外してからプログラムを実行することです。

```bash
env -i HOME=$HOME OPENAI_API_KEY=your_openai_api_key /bin/bash --noprofile --norc

# 次にプログラムを実行
/path/to/your/script/update_targets_infos.sh
```

このターミナルでプログラムを実行することで、`crontab`環境下でのプログラムの動作を確認できます。
