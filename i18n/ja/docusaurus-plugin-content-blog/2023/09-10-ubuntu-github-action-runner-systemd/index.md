---
slug: ubuntu-github-runner-systemd
title: GitHub Runnerの自動実行
authors: Z. Yuan
tags: [github-action, runner]
image: /ja/img/2023/0910.webp
description: UbuntuのSystemdを使ってバックグラウンドで自動実行する方法。
---

GitHubで協力して作業を行う際、プライベートサーバーを使ってCI/CDを行うことがよくあります。

GitHubではこの設定方法についてのドキュメントが提供されており、その手順に従えば簡単に設定が完了します。

<!-- truncate -->

<div align="center">
<figure style={{"width": "80%"}}>
![github_set_runner](./img/github_set_runner.jpg)
</figure>
<figcaption>ドキュメント</figcaption>
</div>

## 問題の説明

しかし、サーバーが再起動されるたびに、もし設定をしていなければ、Runnerサービスは永遠に停止したままになります。そのまま放置しておくと、反応がなくなったり、クレームが来たりするまで、何日も気づかないことがあります。

このようなことが何度も繰り返されて、非常に煩わしいことになります！

そこで、私たちは自動実行が必要です！

## 設定手順

サーバーの起動時に特定のタスクを自動実行させるためには、systemdを使用します。

1. **新しいsystemdサービスファイルを作成する：**

   ```bash
   sudo vim /etc/systemd/system/actions-runner.service
   ```

2. **以下の内容をファイルに貼り付けます：**

   ```bash {7-9} title="/etc/systemd/system/actions-runner.service"
   [Unit]
   Description=GitHub Action Runner
   After=network.target

   [Service]
   Type=simple
   User=あなたのユーザー名
   WorkingDirectory=/home/あなたのユーザー名/actions-runner
   ExecStart=/home/あなたのユーザー名/actions-runner/run.sh
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

   色で強調されている部分に注意してください：

   - `User`、`ExecStart`、`WorkingDirectory`は自分のユーザー名に変更してください。

3. **systemdに新しいサービス設定を読み込ませる：**

   ```bash
   sudo systemctl daemon-reload
   ```

4. **このサービスを有効にし、サーバー起動時に自動的に起動するように設定します：**

   ```bash
   sudo systemctl enable actions-runner.service
   ```

5. **サービスを手動で開始するか、再起動してテストします：**

   ```bash
   sudo systemctl start actions-runner.service
   ```

この方法を使うことで、サーバー起動時に`actions-runner`がバックグラウンドで自動的に実行されます。

サービスを停止したい場合は、以下のコマンドを使用します：

```bash
sudo systemctl stop actions-runner.service
```

:::warning
`run.sh`が実行可能な権限を持っていることを確認してください。
:::

## ステータスの確認

systemdでサービスを管理している場合、ログを確認してその動作状況を把握できます。

以下のコマンドを使用してください：

```bash
sudo journalctl -u actions-runner.service -f
```

パラメータの説明：

- `-u actions-runner.service`：`actions-runner`サービスのログのみを表示します。
- `-f`：このオプションを使用すると、`journalctl`が新しいログをリアルタイムで追跡し、最新の出力を見ることができます。

また、サービスの現在の状態を確認したい場合は、以下のコマンドを使用できます：

```bash
sudo systemctl status actions-runner.service
```

これにより、`actions-runner`サービスの現在の状態や、実行中かどうか、最新のログ出力が表示されます：

<div align="center">
<figure style={{"width": "80%"}}>
![action-service](./img/action-service.jpg)
</figure>
</div>

## 再設定

これは余談ですが、自動実行とは関係ありません。

元々のRunnerが消えてしまった場合（通常はリポジトリをPublicとPrivateに切り替えたときなど）や、Runnerが長期間呼び出されていなかった場合、要するにRunnerを紛失した場合があります！

この場合、再設定が必要です：

1. GitHubアカウントから新しいTokenを取得します。
2. actions-runnerディレクトリ（おそらく自分で設定した別の名前のディレクトリ）に戻り、`.runner`ファイルを削除し、設定コマンドを実行します：

   ```bash
   ./config.sh --url ... (新しいTokenで設定)
   ```

他の手順はそのままで、設定が完了したらサービスを再起動するだけです。