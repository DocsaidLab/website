---
slug: ubuntu-github-runner-systemd
title: GitHub Runner の自動実行
authors: Z. Yuan
tags: [github-action, runner]
image: /ja/img/2023/0910.webp
description: Ubuntu Systemd を使ってバックグラウンドで実行し、自動起動を実現する。
---

GitHub を使用してコラボレーションを行う際、プライベートサーバーを利用して CI/CD を実行することがよくあります。

この部分について、GitHub では初期設定の方法を説明するドキュメントが提供されており、その手順に従えば設定が完了します。

<!-- truncate -->

<div align="center">
<figure style={{"width": "80%"}}>
![github_set_runner](./img/github_set_runner.jpg)
</figure>
<figcaption>ドキュメント</figcaption>
</div>

## 問題の説明

しかし、しばらくして、何らかの理由でサーバーが再起動されると、Runner が自動的に起動しないという問題が発生しました。

この問題は忘れられがちで、CI/CD が機能していないことに気づく頃には数日が経過していることもあります。

こうした問題が繰り返し発生し、非常に困ります。

そのため、自動的に起動する仕組みが必要です！

## 設定手順

サーバー起動後にタスクを自動的に実行するには、systemd を使用する必要があります。

1. **新しい systemd サービスファイルを作成します：**

   ```bash
   sudo vim /etc/systemd/system/actions-runner.service
   ```

2. **以下の内容をファイルに貼り付けます：**

   ```bash {7-9}
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

   特に注意すべき部分：

   - `User`、`ExecStart`、`WorkingDirectory` を自身のユーザー名に変更してください。

3. **systemd に新しいサービス設定を読み込ませます：**

   ```bash
   sudo systemctl daemon-reload
   ```

4. **このサービスを有効にして、起動時に自動実行されるようにします：**

   ```bash
   sudo systemctl enable actions-runner.service
   ```

5. **サービスを手動で起動するか、再起動してテストします：**

   ```bash
   sudo systemctl start actions-runner.service
   ```

この方法を使用すると、サーバーが起動すると同時に actions-runner が自動的にバックグラウンドで実行されます。

サービスを停止したい場合は、次のコマンドを使用します：

```bash
sudo systemctl stop actions-runner.service
```

:::warning
`run.sh` に実行権限があることを確認してください。
:::

## 状態の確認

systemd を使用してサービスを管理する場合、ログを確認して動作状況を把握することができます。

以下のコマンドを使用してください：

```bash
sudo journalctl -u actions-runner.service -f
```

説明：

- `-u actions-runner.service`：actions-runner という名前のサービスのログのみを表示します。
- `-f`：リアルタイムで最新のログを追跡します。

また、サービスの現在の状態を確認するには：

```bash
sudo systemctl status actions-runner.service
```

これにより、`actions-runner` サービスの現在の状態、稼働状況、最近のログ出力が表示されます：

<div align="center">
<figure style={{"width": "80%"}}>
![action-service](./img/action-service.jpg)
</figure>
</div>

## 再設定

もし Runner が失われた場合、通常はリポジトリの公開/非公開の切り替え、または Runner の長期間の非稼働が原因です。この場合、Runner を再設定する必要があります。

その際、actions-runner フォルダー内の `.runner` ファイルを削除し、再度実行してください：

```bash
./config.sh --url ... （新しいトークンで設定）
```

他の手順は同じで、設定が完了したらサービスを再起動すれば問題ありません。
