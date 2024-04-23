---
sidebar_position: 7
---

# 整合 API 功能

在這個章節，我們將介紹如何整合 GmailAPI 和 OpenAI API，讓這兩個功能可以一起運作。

## 主程式

我們先來看一下主程式的架構：

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

    # 調用 OpenAI API
    summary = chatgpt_summary(results)
    summary = f'{summary}\n\n---\n\n以上報告由 OpenAI GPT-3.5 Turbo 模型自動生成。'

    # 生成 Markdown 報告
    markdown_file = generate_markdown_report(
        summary, project_name, current_date.strftime('%Y-%m-%d'))
    print(f"Markdown 文件已生成: {markdown_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate a project update report.")
    parser.add_argument(
        "--project_name",
        type=str,
        help="The name of the project to track.",
        default="Albumentations"
    )
    parser.add_argument(
        "--time_length",
        type=int,
        help="The time length (in days) to track updates.",
        default=1
    )

    args = parser.parse_args()

    main(args.project_name, args.time_length)
```

## 主程式介紹

### 1. 導入模組

程式開頭導入了需要的 Python 模塊和自定義的函數：
- `ArgumentParser` 用於解析命令行參數。
- `datetime` 和 `timedelta` 用於處理日期和時間計算。
- `build_service`, `get_messages`, `parse_message` 是用於處理 Gmail API 的相關操作。
- `chatgpt_summary` 是用來調用 OpenAI API 生成郵件內容的總結。

### 2. 生成報告

`generate_markdown_report` 這個函數負責生成 Markdown 格式的報告文件：
- 接收 `input_text`（郵件內容的總結）、`project_name`（專案名稱）和 `date`（日期）作為參數。
- 將 `input_text` 中的換行符替換為 Markdown 的換行格式。
- 生成 Markdown 文件的標題和內容，然後寫入一個以專案名稱和日期命名的 `.md` 文件。

### 3. 主程式

主程式執行以下步驟：

- 設定當前日期和過去日期（基於 `time_length` 參數計算）。
- 通過 `build_service` 函數構建 Gmail API 服務。
- 調用 `get_messages` 獲取指定日期後和包含指定專案名稱的郵件。
- 使用 `tqdm` 進度條顯示郵件解析進度。
- 解析每個郵件的內容並將其傳送到 OpenAI API 進行總結。
- 將總結的內容添加到 Markdown 文件並生成該文件。

### 4. 命令行參數處理

在輸入區塊中，程式解析兩個命令行參數：
- `--project_name`：用戶指定的專案名稱，默認為 "Albumentations"。
- `--time_length`：跟踪更新的時間長度（天），默認為 1 天。

這些參數允許用戶定制化生成報告的專案名和期間。
