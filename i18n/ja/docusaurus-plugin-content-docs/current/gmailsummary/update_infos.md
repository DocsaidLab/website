---
sidebar_position: 8
---

# 機能統合

前述の説明で、すべての機能の開発が完了しました。次に、これらの機能を統合して、システム全体が後続の処理で自動的に実行できるようにします。

## 完全なプロセス

すべてのステップを統合するために、Bash スクリプト[**update_targets_infos.sh**](https://github.com/DocsaidLab/GmailSummary/blob/main/update_targets_infos.sh)を作成しました。これにより、関数呼び出しと GitHub への自動プッシュが実現されます。

現在、以下のプロジェクトを追跡しています：

- **albumentations**
- **onnxruntime**
- **pytorch-lightning**
- **BentoML**
- **docusaurus**

プロセスにはログ機構も追加されており、後で追跡できるようになっています。

```bash
#!/bin/bash

# ディレクトリと環境変数の定義
origin_dir="$HOME/workspace/GmailSummary"
targ_dir="$HOME/workspace/website"
pyenv_path="$HOME/.pyenv/versions/3.10.14/envs/main/bin/python"
log_dir="$origin_dir/logs"
current_date=$(date '+%Y-%m-%d')
project_path="$targ_dir/docs/gmailsummary/news/$current_date"

# 必要なディレクトリの作成
mkdir -p "$log_dir" "$project_path"

cd $origin_dir

# プロジェクト名リストの指定
project_names=("albumentations" "onnxruntime" "pytorch-lightning" "docusaurus")

for project_name in "${project_names[@]}"; do
    log_file="$log_dir/$project_name-log-$current_date.txt"
    file_name="$project_name.md"

    # Pythonプログラムの実行と出力の処理
    {
        echo "Starting the script for $project_name at $(date)"
        $pyenv_path main.py --project_name $project_name --time_length 1
        mv "$origin_dir/$file_name" "$project_path"
        git -C "$targ_dir" add "$project_path/$file_name"
        echo "Script finished for $project_name at $(date)"
    } >> "$log_file" 2>&1

    # 実行状態の確認
    if [ $? -ne 0 ]; then
        echo "An error occurred for $project_name, please check the log file $log_file." >> "$log_file"
    fi
done

# Gitの変更をプッシュ
git -C "$targ_dir" commit -m "[C] Update project report for $current_date"
git -C "$targ_dir" push
```

## 実装の提案

このプロジェクトでは API を統合するために多くの認証情報やキーが必要ですので、いくつかの提案をします：

まず、絶対に避けるべきことは**認証情報やキーをハードコーディングしないこと**です。

このような方法では、認証情報やキーが漏洩し、メールやデータの安全性が確保できなくなります。

これらの機密情報は安全な場所に保管し、決して公開しないようにしてください。

- **セキュリティの確保**：Gmail API や OpenAI API を扱う際は、`credentials.json`や API キーを適切に保管してください。

その他、少しのアドバイスです：

- **メールの多様性を考慮**：メールをフィルタリングし解析する際、異なるタイプのメールフォーマットや内容に柔軟に対応できるようにプログラムを設計することをお勧めします。
- **定期的なチェックとメンテナンス**：自動化されたシステムであっても、定期的に実行状況を確認し、API の変更に対応するためにプログラムを更新することが重要です。
