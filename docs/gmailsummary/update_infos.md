---
sidebar_position: 8
---

# 串接所有功能

我們已經完成了所有功能的開發，現在我們需要把這些功能串接起來，讓整個系統在後續的運作中能夠自動化執行。

## 完整流程

最後，我把所有的步驟串接在一起，寫成了一個程式 [**update_targets_infos.py**](https://github.com/DocsaidLab/GmailSummary/blob/main/update_targets_infos.sh)。

其中包括了函數調用還有自動推送到 GitHub 的功能。

在現階段，我們追蹤了幾個專案，包含：

- **albumentations**
- **onnxruntime**
- **pytorch-lightning**
- **BentoML**
- **docusaurus**

過程中我們也加入了一些 Log 機制，以便於後續的追蹤。

```bash
#!/bin/bash

cd $HOME/workspace/GmailSummary

# 指定項目名稱列表
project_names=("albumentations" "onnxruntime" "pytorch-lightning" "BentoML" "docusaurus")
log_dir="logs"
news_dir="news"
current_date=$(date '+%Y-%m-%d')

# 創造日誌資料夾，若已存在則忽略
mkdir -p $log_dir

# 創造news目錄，若已存在則忽略
mkdir -p $news_dir

for project_name in "${project_names[@]}"; do

    log_file="$log_dir/$project_name-log-$current_date.txt"

    project_path="$news_dir/$project_name"

    # 開始執行並記錄日誌
    {
        echo "Starting the script for $project_name at $(date)"

        # 執行 Python 程式
        $HOME/.pyenv/versions/3.8.18/envs/main/bin/python main.py --project_name $project_name --time_length 1 2>&1

        # 構造文件名
        file_name="$project_name-update-$current_date.md"

        # 創造專案資料夾，若已存在則忽略
        mkdir -p $project_path
        mv $file_name "$project_path/README.md" 2>&1

        # 將新文件添加到 Git
        git add "$project_path/README.md" 2>&1

        # 提交更改
        git commit -m "[C] Updare $project_name report for $current_date" 2>&1

        # 推送到 GitHub
        git push 2>&1

        echo "Script finished for $project_name at $(date)"

    } >> "$log_file" 2>&1

    # 檢查最後命令是否成功
    if [ $? -ne 0 ]; then
        echo "An error occurred for $project_name, please check the log file $log_file."
    fi

done
```

## 實作建議

在實施這一個自動化方案時，我們有些建議：

首先，不管怎樣，拜託不要：**硬編碼你的憑證和密鑰**。

這樣做會導致你的憑證和密鑰泄露，進而導致你的郵件和數據不再安全。

請將這些敏感信息存儲在安全的地方，並且不要將它們直接公開在任何場合。

- **確保安全性**：處理 Gmail API 和 OpenAI API 時，請妥善保管你的 `credentials.json` 和 API 密鑰。

其他就不是很重要了，就是一些小建議：

- **考慮郵件多樣性**：在過濾和解析郵件時，考慮到不同類型的郵件格式和內容，使程式能夠靈活應對各種情況。
- **定期檢查與維護**：雖然這是一個自動化方案，但定期檢查執行狀況和更新程式以適應可能的 API 變更仍然是必要的。