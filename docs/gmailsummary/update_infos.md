---
sidebar_position: 8
---

# 串接所有功能

經過前面的說明，我們已經完成了所有功能的開發，現在需要把這些功能串接起來，讓整個系統在後續的運作中能夠自動化執行。

## 完整流程

我們撰寫一支 bash 程式 [**update_targets_infos.sh**](https://github.com/DocsaidLab/GmailSummary/blob/main/update_targets_infos.sh)，將所有的步驟串接在一起。

其中包括了函數調用還有自動推送到 GitHub 的功能。

在現階段，我們追蹤了幾個專案，包含：

- **albumentations**
- **onnxruntime**
- **pytorch-lightning**
- **BentoML**
- **docusaurus**

過程中也加入了一些 Log 機制，以便於後續的追蹤。

```bash
#!/bin/bash

# 定義目錄和環境變量
origin_dir="$HOME/workspace/GmailSummary"
targ_dir="$HOME/workspace/website"
pyenv_path="$HOME/.pyenv/versions/3.10.14/envs/main/bin/python"
log_dir="$origin_dir/logs"
current_date=$(date '+%Y-%m-%d')
news_dir="$targ_dir/docs/gmailsummary/news/$current_date"

# 創建所需的目錄
mkdir -p "$log_dir" "$news_dir"

# 指定項目名稱列表
project_names=("albumentations" "onnxruntime" "pytorch-lightning" "BentoML" "docusaurus")

for project_name in "${project_names[@]}"; do
    log_file="$log_dir/$project_name-log-$current_date.txt"
    project_path="$news_dir/$project_name"
    file_name="$project_name.md"
    mkdir -p "$project_path"

    # 執行 Python 程式並處理輸出
    {
        echo "Starting the script for $project_name at $(date)"
        $pyenv_path main.py --project_name $project_name --time_length 1
        mv "$origin_dir/$file_name" "$project_path"
        git -C "$targ_dir" add "$project_path/$file_name"
        echo "Script finished for $project_name at $(date)"
    } >> "$log_file" 2>&1

    # 檢查執行狀態
    if [ $? -ne 0 ]; then
        echo "An error occurred for $project_name, please check the log file $log_file." >> "$log_file"
    fi
done

# 推送 Git 變更
{
    git -C "$targ_dir" commit -m "[C] Update project report for $current_date"
    git -C "$targ_dir" push
} >> "$log_file" 2>&1
```

## 實作建議

在這個專案中，為了串接 API 而使整個專案充斥著憑證和密鑰，因此我們有些建議：

首先，不管怎樣，拜託不要：**硬編碼你的憑證和密鑰**。

這樣做會導致你的憑證和密鑰泄露，進而導致你的郵件和數據不再安全。

請將這些敏感信息存儲在安全的地方，並且不要將它們直接公開在任何場合。

- **確保安全性**：處理 Gmail API 和 OpenAI API 時，請妥善保管你的 `credentials.json` 和 API 密鑰。

其他就不是很重要了，就是一些小建議：

- **考慮郵件多樣性**：在過濾和解析郵件時，考慮到不同類型的郵件格式和內容，使程式能夠靈活應對各種情況。
- **定期檢查與維護**：雖然這是一個自動化方案，但定期檢查執行狀況和更新程式以適應可能的 API 變更仍然是必要的。