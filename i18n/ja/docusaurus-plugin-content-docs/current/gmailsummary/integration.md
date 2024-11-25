---
sidebar_position: 7
---

# API 機能の統合

この章では、Gmail API と OpenAI API をどのように統合して、これらの機能が一緒に動作するようにするかを紹介します。

## メインプログラム

まず、メインプログラムの構成を見てみましょう：

```python
from argparse import ArgumentParser
from datetime import datetime, timedelta

from gmail_api import build_service, get_messages, parse_message
from openai_api import chatgpt_summary
from tqdm import tqdm


def generate_markdown_report(input_text, project_name, date):
    title = f"{project_name} 更新報告 - {date}"
    markdown_text = input_text.replace('\n', '\n\n')
    markdown_content = f"# {title}\n\n{markdown_text}"
    file_name = f"{project_name}-update-{date}.md"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(markdown_content)
    return file_name


def main(project_name, time_length):

    current_date = datetime.now()
    after_date = current_date - timedelta(days=time_length)
    after_date = after_date.strftime('%Y/%m/%d')

    service = build_service()
    messages = get_messages(
        service,
        after_date=after_date,
        subject_filter=project_name
    )

    results = [
        parse_message(service, msg['id'])
        for msg in tqdm(messages)
    ]

    # OpenAI APIの呼び出し
    summary = chatgpt_summary(results)
    summary = f'{summary}\n\n---\n\n以上の報告はOpenAI GPT-3.5 Turboモデルによって自動生成されました。'

    # Markdown報告書の生成
    markdown_file = generate_markdown_report(
        summary, project_name, current_date.strftime('%Y-%m-%d'))
    print(f"Markdownファイルが生成されました: {markdown_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="プロジェクトの更新報告を生成します。")
    parser.add_argument(
        "--project_name",
        type=str,
        help="追跡するプロジェクト名。",
        default="Albumentations"
    )
    parser.add_argument(
        "--time_length",
        type=int,
        help="更新を追跡する期間（日単位）。",
        default=1
    )

    args = parser.parse_args()

    main(args.project_name, args.time_length)
```

## メインプログラムの説明

### 1. モジュールのインポート

プログラムの冒頭で必要な Python モジュールとカスタム関数をインポートしています：

- `ArgumentParser` はコマンドライン引数を解析するために使用します。
- `datetime` と `timedelta` は日付と時間の計算を処理するために使用します。
- `build_service`, `get_messages`, `parse_message` は Gmail API に関連する操作を行います。
- `chatgpt_summary` は OpenAI API を呼び出してメール内容の要約を生成するために使用します。

### 2. 報告書の生成

`generate_markdown_report` 関数は Markdown 形式の報告書を生成します：

- `input_text`（メール内容の要約）、`project_name`（プロジェクト名）、`date`（日付）の 3 つの引数を受け取ります。
- `input_text`内の改行を Markdown 形式の改行に置き換えます。
- プロジェクト名と日付を使用して、報告書のタイトルと内容を作成し、それを指定された`.md`ファイルに書き込みます。

### 3. メインプログラム

メインプログラムは以下の手順で実行されます：

- 現在の日付と過去の日付（`time_length`パラメータに基づく）を設定します。
- `build_service`関数を使用して Gmail API サービスを構築します。
- `get_messages`を呼び出して、指定された日付以降に特定のプロジェクト名を含むメールを取得します。
- `tqdm`の進捗バーを使って、メール解析の進行状況を表示します。
- 各メールの内容を解析し、その情報を OpenAI API に送信して要約を取得します。
- 要約を Markdown ファイルに追加し、そのファイルを生成します。

### 4. コマンドライン引数の処理

プログラムは 2 つのコマンドライン引数を解析します：

- `--project_name`：ユーザーが指定したプロジェクト名（デフォルトは "Albumentations"）。
- `--time_length`：更新を追跡する期間（日数）（デフォルトは 1 日）。

これらの引数により、ユーザーは報告書のプロジェクト名と期間をカスタマイズできます。
