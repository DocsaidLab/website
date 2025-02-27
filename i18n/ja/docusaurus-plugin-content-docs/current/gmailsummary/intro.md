---
sidebar_position: 1
---

# イントロダクション

日常生活の中で、私たちはよく GitHub のあるプロジェクトで「Watch」オプションをクリックすることによって、そのプロジェクトの活動更新メールを受け取るようになります。これらの更新には、新機能の議論、issue の報告、pull request (PR)、バグ報告などが含まれます。

例えば、いくつかの GitHub プロジェクトをフォローして「すべてのアクティビティ」の設定を採用した場合：

- [**albumentations**](https://github.com/albumentations-team/albumentations): 約毎日 10 通のメール。
- [**onnxruntime**](https://github.com/microsoft/onnxruntime): 約毎日 200 通のメール。
- [**PyTorch**](https://github.com/pytorch/pytorch): 約毎日 1,500 通のメール。

もしさらに多くのプロジェクトをフォローしていたら、毎日 5,000 通以上のメールを受け取ることになるでしょう。

＊

**本当に誰かが「1 通も見逃さずに」これらのメールを読むのでしょうか？**

＊

私は絶対に読まないですね、通常は開かずに削除してしまいます。

したがって、効率（手抜き）を追求するエンジニアとして、この問題をどう解決するかを考えなければなりません。

## 問題の分解

大量のメールの問題を解決するために、問題を 2 つの部分に分けて考えます：自動ダウンロードと自動分析。

### 自動ダウンロード

Gmail からこれらのメールを自動的にダウンロードし、重要な情報を抽出する必要があります。

考えられる解決策は以下の通りです：

1. **Zapier や IFTTT などのサービスを使用**

   - [**Zapier**](https://zapier.com/)：これは作業効率を高めるための自動化プラットフォームで、Gmail、Slack、Mailchimp など、3,000 以上の異なる Web アプリケーションを接続することができます。このプラットフォームでは、ユーザーが自動化ワークフローを作成し、異なるアプリケーション間で自動的にインタラクションを実現できます。
   - [**IFTTT**](https://ifttt.com/)：IFTTT は無料の Web サービスで、ユーザーが「これをしたら、あれをする」という自動化タスクを作成することを許可します。これらのタスクは「Applets」と呼ばれ、各 Applet は 1 つのトリガーと 1 つのアクションで構成されています。トリガー条件が満たされると、Applet は自動的にアクションを実行します。

2. **GmailAPI を使用**

   - [**GmailAPI**](https://developers.google.com/gmail/api)：この API を使うと、プログラムでメールを読む、送る、検索するなどの操作ができます。

:::tip
プログラムを書くなら、最初の方法は考えなくてもいいので、GmailAPI を使いましょう。
:::

### 自動分析

大量のメールを取得した後、これらのメールを分析して重要な情報を見つけ出す必要があります。

この部分は、今の ChatGPT の時代ではそれほど難しくはありません。ChatGPT を使って自然言語処理を行い、メール内の重要な情報を抽出できます。

## 最後に

プロセス全体をいくつかの部分に分けて説明します：

1. **メールの自動ダウンロード**：GmailAPI を使用。
2. **メール内容の解析**：ロジックを自作。
3. **メール内容の要約**：ChatGPT を使用。
4. **出力＆スケジューリング**：Markdown で出力し、crontab でスケジュール。

以上が本プロジェクトの核心機能です。成果は**出力サンプル**ページで展示されています。

次に、これらの各部分の機能実装方法について順次説明していきます。
