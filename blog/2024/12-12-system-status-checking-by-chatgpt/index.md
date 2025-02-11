---
slug: system-status-checking-by-chatgpt
title: Ubuntu 系統基礎狀態檢查自動化
authors: Z. Yuan
image: /img/2024/1212.webp
tags: [System-Monitoring, Automation, OpenAI-Integration]
description: 透過 ChatGPT 自動化檢查系統基礎狀態。
---

為了分析問題，我們需要檢查系統狀態，但是又不擅長閱讀系統日誌。

這時候我們可以透過 ChatGPT 來自動化檢查系統基礎狀態。

<!-- truncate -->

## 功能重點

首先透過內建 Linux 指令如 `uptime`、`top`、`free`、`df`、`systemctl`、`sensors`、`smartctl`、`dmesg`、`journalctl`、`ss`、`ip`、`docker` 等收集系統相關狀況與性能指標。

接著我們將原始報告分為數個區塊，以 OpenAI API 為後端的 GPT 模型 (指定為 `gpt-4o`) 自動生成摘要並彙整。

最後將多份摘要整合，最後生成一份 Markdown 格式的系統狀況報告，包括：

- 系統現況描述
- 系統問題分析
- 改善與建議方案

:::warning
本篇文章的程式碼可能會隨著 OpenAI 的 API 更新而需要調整，請確保 API 的使用方式與程式碼相符。
:::

## 前置作業

1. **必要套件與指令**：

   - `curl`：用於與 API 進行 HTTP 通訊
   - `jq`：用於解析 JSON 格式的 API 回應
   - `smartctl`：用於檢查磁碟 SMART 資訊
   - `systemctl`：檢視服務執行與失敗狀態
   - `docker`：檢查 Docker 容器狀態（若系統未使用 Docker，可忽略錯誤訊息）
   - `sensors`：取得系統溫度資訊（若無此指令則無法取得溫度，程式會顯示 Warning）

   請確保以上指令已正確安裝並可在該系統正常執行。

2. **OpenAI API Key**：

   本程式會使用 OpenAI API 產生系統報告摘要與最終分析。

   請確保您已擁有有效的 OpenAI API 金鑰，並將此金鑰儲存於指定路徑

   - 預設為 `/home/your_user_name/.openai_api_key`

   ```bash
   echo "YOUR_OPENAI_API_KEY" > /home/your_user_name/.openai_api_key
   chmod 600 /home/your_user_name/.openai_api_key
   ```

   確保將 `YOUR_OPENAI_API_KEY` 替換為實際的 API Key。

3. **工作目錄與磁碟空間**：

   預設工作目錄為 `/var/log/system_checking_by_chatgpt`，程式會在此建立相關檔案與資料夾：

   - `raw_report_YYYY-MM-DD.txt`：原始資訊收集報告
   - `chunks_YYYY-MM-DD/`：分塊後的檔案
   - `summary_YYYY-MM-DD/`：各分塊經 GPT 摘要後的檔案
   - `combined_summary_YYYY-MM-DD.txt`：合併所有分塊摘要的最終摘要檔
   - `final_report_YYYY-MM-DD.md`：最終分析報告（Markdown 格式）
   - `debug_YYYY-MM-DD.log`：除錯日誌檔

   請確保該目錄擁有足夠空間（至少 100MB）。預設會檢查空間不足時自動退出。

4. **執行權限**：

   請確保此程式擁有可執行權限：

   ```bash
   chmod +x system_checking_by_chatgpt.sh
   ```

## 執行方式

在終端機中執行本程式即可開始產生報告。

預期的執行流程如下：

1. **初始化設定與檢查**：程式會檢查所有必要的外部指令是否存在並確認 `OPENAI_API_KEY` 不為空。若有問題會立即中斷並顯示錯誤。
2. **系統資訊收集**：程式透過各種指令收集系統資訊並將輸出匯整至 `raw_report_YYYY-MM-DD.txt`。
3. **分塊處理與摘要**：程式將 `raw_report_YYYY-MM-DD.txt` 分成多個約 200 行大小的子檔案 (`chunk_*`)，每個子檔經由 OpenAI API 處理生成摘要檔 (`summary_chunk_*`)，以提煉該分塊關鍵重點與問題。
4. **合併所有摘要並最終分析**：程式將所有摘要彙整為單一文件 `combined_summary_YYYY-MM-DD.txt`，再呼叫 OpenAI API 進行「最終報告」的生成。
5. **產出最終報告**：最終報告儲存在 `final_report_YYYY-MM-DD.md`，內含：
   - 系統現況描述
   - 問題與異常事件分析
   - 建議的解決方案與檢測建議（可能含表格與指令範例）

## 注意事項

1. **API 金鑰安全性**：請妥善保管 `OPENAI_API_KEY`，此金鑰不應公開，並建議限制檔案的讀取權限（如 600 權限）。
2. **智慧調整檢測範圍**：若希望擴充或縮減蒐集的系統資訊，可在 `collect_system_info()` 函數中新增或移除指令。
3. **OpenAI API 模型與 Token 限制**：預設使用 `MODEL="gpt-4o"`，請確保此模型在您的 API 權限中可用。若摘要或最終報告過長而導致 API 回應失敗或中斷，請考慮調整 `CHUNK_SIZE` 或降低輸出內容。
4. **時區與日期格式**：程式使用 `date +"%Y-%m-%d"` 格式化日期，可依需求修改。

## 範例流程

以下為完整運行範例說明：

1. **確認環境與 API Key**：

   ```bash
   echo "sk-abc123xxx..." > /home/your_user_name/.openai_api_key
   chmod 600 /home/your_user_name/.openai_api_key
   ```

2. **執行程式**：

   ```bash
   sudo bash system_checking_by_chatgpt.sh
   ```

   程式會顯示：

   ```
   [INFO] 開始執行程式...
   [INFO] API 金鑰已成功載入。
   [INFO] 開始收集系統資訊...
   [INFO] 系統資訊收集完成，儲存於 /var/log/system_checking_by_chatgpt/raw_report_2024-12-12.txt
   [INFO] 對 /var/log/system_checking_by_chatgpt/chunks_2024-12-12/chunk_aa 進行摘要請求...
   ...
   [INFO] 最終報告已生成: /var/log/system_checking_by_chatgpt/final_report_2024-12-12.md
   [INFO] 程式執行完成。
   ```

3. **檢視報告**：

   ```bash
   less /var/log/system_checking_by_chatgpt/final_report_2024-12-12.md
   ```

4. **檢閱除錯日誌（若需要）**：
   ```bash
   less /var/log/system_checking_by_chatgpt/debug_2024-12-12.log
   ```

## 程式碼

```shell title="system_checking_by_chatgpt.sh"
#!/usr/bin/env bash

set -euo pipefail
# set -x  # 如需除錯可啟用

########################################
# 基本設定與檢查
########################################

DATE=$(date +"%Y-%m-%d")
WORK_DIR="/var/log/system_checking_by_chatgpt"
RAW_REPORT="$WORK_DIR/raw_report_$DATE.txt"
CHUNKS_DIR="$WORK_DIR/chunks_$DATE"
SUMMARY_DIR="$WORK_DIR/summary_$DATE"
FINAL_REPORT="$WORK_DIR/final_report_$DATE.md"  # 使用 Markdown 格式
DEBUG_LOG="$WORK_DIR/debug_$DATE.log"

OPENAI_KEY_FILE="/home/your_user_name/.openai_api_key"

mkdir -p "$WORK_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p "$SUMMARY_DIR"

# 檢查必要指令
REQUIRED_COMMANDS=("curl" "jq" "smartctl" "systemctl" "docker" "sensors")
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[ERROR] 必須的指令 $cmd 未安裝，請先安裝。" | tee -a "$DEBUG_LOG"
        exit 1
    fi
done

# 檢查磁碟空間
AVAILABLE_SPACE=$(df "$WORK_DIR" | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 102400 ]; then # 少於 100MB
    echo "[ERROR] $WORK_DIR 磁碟空間不足。" | tee -a "$DEBUG_LOG"
    exit 1
fi

# 日誌函數：將訊息輸出到stderr，以免干擾函數回傳值
log() {
    local LEVEL=$1
    shift
    local MESSAGE="$@"
    # 將訊息寫入DEBUG_LOG並透過stderr顯示
    {
        echo "[$LEVEL] $MESSAGE" | tee -a "$DEBUG_LOG" >&2
        logger -t "system_checking_by_chatgpt" "[$LEVEL] $MESSAGE"
    }
}

log "INFO" "開始執行程式..."

########################################
# 讀取 API 金鑰
########################################

if [ ! -f "$OPENAI_KEY_FILE" ]; then
    log "ERROR" "$OPENAI_KEY_FILE 不存在。"
    exit 1
fi

OPENAI_API_KEY=$(cat "$OPENAI_KEY_FILE")
if [ -z "$OPENAI_API_KEY" ]; then
    log "ERROR" "OPENAI_API_KEY 為空值。"
    exit 1
fi
export OPENAI_API_KEY

log "INFO" "API 金鑰已成功載入。"

API_URL="https://api.openai.com/v1/chat/completions"
MODEL="gpt-4o"

########################################
# 函數定義區
########################################

collect_system_info() {
    # 收集系統資訊到 RAW_REPORT
    {
        echo "=== System Uptime ==="
        uptime
        echo ""

        echo "=== Date & Time ==="
        date
        echo ""

        echo "=== CPU & Memory Usage ==="
        top -b -n1 | head -n 20
        echo ""

        echo "=== Memory Usage (free) ==="
        free -h
        echo ""

        echo "=== Disk Usage (df) ==="
        df -h
        echo ""

        echo "=== Failed Services ==="
        systemctl list-units --state=failed
        echo ""

        echo "=== Temperature Sensors ==="
        sensors 2>/dev/null || echo "[WARNING] 無法取得溫度資訊"
        echo ""

        echo "=== NVMe / SMART Status ==="
        sudo smartctl -a /dev/nvme0n1 2>/dev/null || echo "[WARNING] 無法檢查 /dev/nvme0n1"
        echo ""
        sudo smartctl -a /dev/nvme1n1 2>/dev/null || echo "[WARNING] 無法檢查 /dev/nvme1n1"
        echo ""

        echo "=== Recent dmesg Entries (Last 100 lines) ==="
        dmesg | tail -n 100
        echo ""

        echo "=== System Journal (Last 300 lines, errors only) ==="
        journalctl -p err -n 300
        echo ""

        echo "=== Network Status (ss -tulpn) ==="
        ss -tulpn 2>/dev/null || echo "[WARNING] 無法檢查網路狀態"
        echo ""

        echo "=== Network Interface Statistics (ip -s link) ==="
        ip -s link
        echo ""

        echo "=== Docker Containers (if any) ==="
        docker ps -a 2>/dev/null || echo "[INFO] Docker 未運行或未安裝"
        echo ""

    } > "$RAW_REPORT"
}

generate_chunk_summaries() {
    # 將 RAW_REPORT 分塊並摘要
    CHUNK_SIZE=200
    split -l $CHUNK_SIZE "$RAW_REPORT" "$CHUNKS_DIR/chunk_"

    CHUNKS=("$CHUNKS_DIR"/chunk_*)

    if [ ${#CHUNKS[@]} -eq 0 ]; then
        log "ERROR" "沒有產生任何 chunk！"
        exit 1
    fi

    # 分塊摘要提示
    CHUNK_SYSTEM_PROMPT="你是一位專業的系統管理顧問。以上是系統資訊的一部分，請分析：1.重要的系統狀態資訊（如 CPU/記憶體使用率、磁碟使用、網路狀態、服務運行和錯誤訊息），2.任何顯示異常、錯誤或需要特別注意的地方。該系統可能遭到惡意攻擊，請先分析這裡檢查的目標，接著詳細描述你看到的問題，以及問題所對應的錯誤或警告訊息。"

    summaries=()

    for chunk_file in "${CHUNKS[@]}"; do
        CHUNK_CONTENT=$(cat "$chunk_file")

        CHUNK_API_PAYLOAD=$(jq -n \
            --arg system_prompt "$CHUNK_SYSTEM_PROMPT" \
            --arg user_prompt "$CHUNK_CONTENT" \
            --arg model "$MODEL" \
            '{
                "model": $model,
                "messages": [
                    {"role": "user", "content": $user_prompt},
                    {"role": "system", "content": $system_prompt}
                ]
            }')

        log "INFO" "對 $chunk_file 進行摘要請求..."
        START_TIME=$(date +%s)
        RESPONSE=$(curl -sS -X POST "$API_URL" \
          -H "Authorization: Bearer $OPENAI_API_KEY" \
          -H "Content-Type: application/json" \
          -d "$CHUNK_API_PAYLOAD")

        if [ $? -ne 0 ]; then
            log "ERROR" "與 ChatGPT API 通訊失敗 (在摘要階段): $chunk_file"
            exit 1
        fi
        END_TIME=$(date +%s)
        log "INFO" "API 請求耗時 $((END_TIME - START_TIME)) 秒。"

        SUMMARY=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>>"$DEBUG_LOG" || echo "")

        if [ -z "$SUMMARY" ] || [ "$SUMMARY" = "null" ]; then
            log "ERROR" "無法取得該分塊的摘要：$chunk_file"
            echo "Raw Response: $RESPONSE" >> "$DEBUG_LOG"
            exit 1
        fi

        SUMMARY_FILE="$SUMMARY_DIR/summary_$(basename "$chunk_file").txt"
        echo "$SUMMARY" > "$SUMMARY_FILE"
        summaries+=("$SUMMARY_FILE")
    done

    # 僅輸出 summary 檔案清單到stdout（不輸出其他日誌），確保呼叫者可正確取得檔名列表
    for s in "${summaries[@]}"; do
        echo "$s"
    done
}

combine_summaries() {
    local summaries=("$@")
    COMBINED_SUMMARY="$WORK_DIR/combined_summary_$DATE.txt"
    rm -f "$COMBINED_SUMMARY"
    touch "$COMBINED_SUMMARY"

    log "INFO" "開始彙整所有分塊摘要..."
    {
        echo "以下為多個分段摘要的合併結果："
        echo "------------------------------------"
        for sfile in "${summaries[@]}"; do
            echo "=== 分段摘要 ==="
            cat "$sfile"
            echo ""
        done
    } > "$COMBINED_SUMMARY"

    echo "$COMBINED_SUMMARY"
}

generate_final_report() {
    local COMBINED_SUMMARY="$1"

    FINAL_SYSTEM_PROMPT="你是一位專業的系統管理顧問。以上是由多個分段摘要組成的合併內容，該系統可能遭到惡意攻擊，請針對整體系統狀況產出「詳細的最終報告」。並包含以下三個主要章節：

**1. 系統現況描述**：
列出目前系統的狀況、服務運行情形、網路狀態、重要日誌資訊、錯誤訊息及任何異常事件的細節。

**2. 系統現況分析**：
針對上述資訊進行分析，指出可能的問題點、異常原因推測、系統效能瓶頸，以及任何不尋常的狀況或風險。

**3. 建議解決方式**：
根據分析結果，提出可行且明確的改善方案、診斷指令或檢查流程建議，包括優化效能、修正錯誤、強化系統穩定性的建議。

在分析過程中，你必須針對每個問題點補上對應的檢查指令和流程建議，必要時可以繪製詳細表格進行記錄。"

    FINAL_INPUT=$(cat "$COMBINED_SUMMARY")

    FINAL_API_PAYLOAD=$(jq -n \
        --arg system_prompt "$FINAL_SYSTEM_PROMPT" \
        --arg user_prompt "$FINAL_INPUT" \
        --arg model "$MODEL" \
        '{
            "model": $model,
            "messages": [
                {"role": "user", "content": $user_prompt},
                {"role": "system", "content": $system_prompt}
            ]
        }')

    log "INFO" "開始最終分析..."

    START_TIME=$(date +%s)
    RESPONSE=$(curl -sS -X POST "$API_URL" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d "$FINAL_API_PAYLOAD")

    if [ $? -ne 0 ]; then
        log "ERROR" "與 ChatGPT API 通訊失敗 (最終分析階段)"
        exit 1
    fi
    END_TIME=$(date +%s)
    log "INFO" "API 請求耗時 $((END_TIME - START_TIME)) 秒。"

    log "DEBUG" "最終分析回應: $RESPONSE"

    FINAL_ANALYSIS=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>>"$DEBUG_LOG" || echo "")

    if [ -z "$FINAL_ANALYSIS" ] || [ "$FINAL_ANALYSIS" = "null" ]; then
        log "ERROR" "最終分析內容取得失敗。"
        echo "Raw Response: $RESPONSE" >> "$DEBUG_LOG"
        exit 1
    fi

    {
        echo "# 每日系統檢測報告 - $DATE"
        echo ""
        echo "$FINAL_ANALYSIS"
    } > "$FINAL_REPORT"

    log "INFO" "最終報告已生成: $FINAL_REPORT"
}


########################################
# 主程式流程
########################################

log "INFO" "開始收集系統資訊..."
collect_system_info
log "INFO" "系統資訊收集完成，儲存於 $RAW_REPORT"

# 使用 readarray 讀取 generate_chunk_summaries 的輸出至陣列
mapfile -t summaries < <(generate_chunk_summaries)
COMBINED_SUMMARY_FILE=$(combine_summaries "${summaries[@]}")
generate_final_report "$COMBINED_SUMMARY_FILE"

log "INFO" "程式執行完成。"
```

## 執行結果展示

```markdown
# 每日系統檢測報告 - 2024-12-12

## 詳細的最終報告

### 1. 系統現況描述

**CPU 使用狀況**：

- CPU 使用率偏高，特別是在處理‘pt_data+’任務時，佔用超過 200%到 300%。
- 多個 CPU 核心的溫度達到甚至超過 80°C，可能意味著過熱問題。

**記憶體使用狀況**：

- 總記憶體 128577.3 MiB，已被使用 48421.6 MiB，有 76743.1 MiB 的緩存/緩衝。
- 系統出現多次「Out of memory」事件，部分進程因內存不足被強制終止。

**磁碟使用狀況**：

- /（根目錄）已使用 70%，/dev/nvme1n1 使用了 51%。
- NVMe 磁碟健康狀況顯示“PASSED”，但有一些 "Invalid Field in Command" 錯誤。

**網路狀態**：

- 網卡`enp5s0`和`r8169`出現多次「NETDEV WATCHDOG」警告，提示網路傳輸隊列超時。
- 日誌顯示持續的 UFW 防火牆封鎖來自 192.168.0.1 的多播封包。
- 存在一次 SSH 連線失敗的記錄。

**服務運行狀態**：

- `snap.firmware-updater.firmware-notifier`服務多次啟動失敗。
- `NETDEV WATCHDOG`錯誤記錄多次。
- 部分應用程序如 `NetworkManager-dispatcher` 服務未能成功啟動。

**安全狀況**：

- 日誌顯示有多重 `apparmor` 拒絕警告，需要檢查安全設定。

### 2. 系統現況分析

**CPU 過熱及高使用率**：

- 高 CPU 使用及溫度可能由於高強度計算任務（如‘pt_data+’），需要檢查是否為必要進程或優化計算。

**記憶體瓶頸**：

- 多次 OOM 事件表明內存的分配不合理或不足，需進一步檢查內存密集型應用程序及其運行狀況。

**磁碟健康及錯誤**：

- NVMe 磁碟出現"Invalid Field in Command"可能是由於驅動或韌體問題，需確認是否有更新。

**網路問題**：

- 「NETDEV WATCHDOG」提示網絡驅動或配置問題，用戶必須排查網卡驅動及配置更新。
- 持續的 UFW 封鎖事件可能由非正常流量或配置錯誤引發，需要檢查該 IP 設備。

**服務運行問題**：

- `snap.firmware-updater.firmware-notifier`服務的問題可能是由於資源不可用或相依服務嚴重故障。

**安全風險**：

- UFW 封鎖與`apparmor`警告顯示潛在安全風險，需檢查防火牆和應用安全性。

### 3. 建議解決方式

**改進 CPU 及記憶體效能**：

1. **檢查并優化進程**：

   - 使用`top`或`htop`監控高 CPU 使用的進程。
   - 調整或關閉非必要的‘pt_data+’進程。

2. **內存管理**：
   - 使用`free -m`和`vmstat`等命令進行內存狀況分析。
   - 增加物理內存或調整 swap 空間。

**修復磁碟錯誤**：

1. **檢查更新驅動/韌體**：
   - 執行`sudo smartctl -a /dev/nvme0n1`檢查詳細 SMART 信息。
   - 更新 NVMe 驅動程序和韌體。

**處理網路問題**：

1. **網卡驅動檢查**：

   - 用`sudo ethtool -i r8169`檢查驅動版本，確認是否有可用升級。
   - `journalctl -k | grep -i netdev`查看更詳細的網卡錯誤日誌。

2. **檢查 UFW 設定**：
   - 使用命令`sudo ufw status verbose`檢查活動的防火牆規則。
   - 具體分析來自 192.168.0.1 的流量來源，確保正當。

**服務及安全性修正**：

1. **服務診斷**：

   - 查看`snap.service`相關的日誌，使用`journalctl -xe`獲得更多錯誤資訊。
   - 檢查`firmware-updater`的相依包狀況和配置。

2. **加強安全配置**：
   - 從`cat /var/log/syslog | grep apparmor`檢查具體的`apparmor`警告信息。
   - 鞏固防火牆及加強網絡服務的安全策略。

最後，建議整個系統進行詳細日誌監控，加強對異常事件快速反應的能力，以保障系統的穩定運行。
```
