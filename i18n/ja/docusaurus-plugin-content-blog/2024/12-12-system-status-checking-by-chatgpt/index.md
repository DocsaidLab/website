---
slug: system-status-checking-by-chatgpt
title: Ubuntu システムの基本状態チェック自動化
authors: Z. Yuan
image: /ja/img/2024/1212.webp
tags: [System-Monitoring, Automation, OpenAI-Integration]
description: ChatGPTを使ってシステムの基本状態を自動でチェックする方法。
---

問題を分析するためにはシステムの状態を確認する必要がありますが、システムログを読むのが得意ではないことがあります。

そのような時、ChatGPT を使ってシステムの基本状態を自動でチェックできます。

<!-- truncate -->

## 機能のポイント

まず、`uptime`、`top`、`free`、`df`、`systemctl`、`sensors`、`smartctl`、`dmesg`、`journalctl`、`ss`、`ip`、`docker` などの組み込み Linux コマンドを使用して、システムの状態とパフォーマンス指標を収集します。

次に、収集したレポートをいくつかのセクションに分け、OpenAI API をバックエンドに使用した GPT モデル（`gpt-4o`として指定）で自動的に要約を生成し、まとめます。

最後に、複数の要約を統合し、最終的に Markdown 形式のシステム状態レポートを生成します。このレポートには以下が含まれます：

- システムの現状説明
- システムの問題分析
- 改善と提案策

:::warning
この記事のコードは OpenAI API の更新により調整が必要になる可能性があります。API の使用方法とコードが一致していることを確認してください。
:::

## 前提作業

1. **必要なパッケージとコマンド**：

   - `curl`：API との HTTP 通信に使用
   - `jq`：JSON 形式の API レスポンスの解析に使用
   - `smartctl`：ディスクの SMART 情報をチェックするために使用
   - `systemctl`：サービスの実行と失敗状態を確認
   - `docker`：Docker コンテナの状態を確認（システムで Docker を使用していない場合はエラーメッセージを無視できます）
   - `sensors`：システムの温度情報を取得（このコマンドがない場合、温度を取得できません。プログラムは警告を表示します）

   上記のコマンドが正しくインストールされ、システムで正常に実行できることを確認してください。

2. **OpenAI API キー**：

   このプログラムは、システムレポートの要約と最終分析のために OpenAI API を使用します。

   有効な OpenAI API キーを所持していることを確認し、そのキーを指定されたパスに保存してください。

   - デフォルトは `/home/your_user_name/.openai_api_key`

   ```bash
   echo "YOUR_OPENAI_API_KEY" > /home/your_user_name/.openai_api_key
   chmod 600 /home/your_user_name/.openai_api_key
   ```

   `YOUR_OPENAI_API_KEY` を実際の API キーに置き換えてください。

3. **作業ディレクトリとディスクスペース**：

   デフォルトの作業ディレクトリは `/var/log/system_checking_by_chatgpt` で、プログラムはこの場所に関連するファイルとフォルダを作成します：

   - `raw_report_YYYY-MM-DD.txt`：原始情報収集レポート
   - `chunks_YYYY-MM-DD/`：分割後のファイル
   - `summary_YYYY-MM-DD/`：GPT によって要約された分割ファイル
   - `combined_summary_YYYY-MM-DD.txt`：すべての要約を統合した最終的な要約ファイル
   - `final_report_YYYY-MM-DD.md`：最終分析レポート（Markdown 形式）
   - `debug_YYYY-MM-DD.log`：デバッグログファイル

   このディレクトリには十分な空き容量（少なくとも 100MB）があることを確認してください。空き容量が不足している場合、プログラムは自動的に終了します。

4. **実行権限**：

   このプログラムには実行権限が必要です：

   ```bash
   chmod +x system_checking_by_chatgpt.sh
   ```

## 実行方法

ターミナルでこのプログラムを実行すると、レポートが生成されます。

予想される実行の流れは以下の通りです：

1. **初期設定とチェック**：プログラムは必要な外部コマンドが存在するか、`OPENAI_API_KEY`が空でないかを確認します。問題がある場合は、直ちに中断しエラーメッセージを表示します。
2. **システム情報の収集**：プログラムは様々なコマンドを使用してシステム情報を収集し、その出力を `raw_report_YYYY-MM-DD.txt` にまとめます。
3. **分割処理と要約**：プログラムは `raw_report_YYYY-MM-DD.txt` を約 200 行ごとのサブファイル（`chunk_*`）に分割し、それぞれのサブファイルを OpenAI API で処理して要約ファイル（`summary_chunk_*`）を生成します。これにより、各分割の重要なポイントと問題を抽出します。
4. **すべての要約を統合し最終分析**：プログラムはすべての要約を 1 つのファイル `combined_summary_YYYY-MM-DD.txt` に統合し、その後 OpenAI API を呼び出して「最終レポート」を生成します。
5. **最終レポートの出力**：最終レポートは `final_report_YYYY-MM-DD.md` に保存され、次の内容が含まれます：
   - システムの現状説明
   - 問題と異常事象の分析
   - 提案される解決策と検証提案（表やコマンドの例を含むことがあります）

## 注意事項

1. **API キーの安全性**：`OPENAI_API_KEY`は適切に保管し、このキーは公開しないでください。ファイルの読み取り権限（600 権限）を制限することをお勧めします。
2. **情報収集の範囲の調整**：収集するシステム情報を拡張または縮小したい場合は、`collect_system_info()` 関数内でコマンドを追加または削除できます。
3. **OpenAI API モデルとトークンの制限**：デフォルトでは `MODEL="gpt-4o"` を使用しています。このモデルが API 権限内で使用できることを確認してください。要約や最終報告が長すぎて API の応答が失敗または中断される場合、`CHUNK_SIZE`を調整するか、出力内容を減らすことを検討してください。
4. **タイムゾーンと日付形式**：プログラムは `date +"%Y-%m-%d"` で日付をフォーマットしますが、必要に応じて変更できます。

## サンプルフロー

以下はプログラムの実行例です：

1. **環境と API キーの確認**：

   ```bash
   echo "sk-abc123xxx..." > /home/your_user_name/.openai_api_key
   chmod 600 /home/your_user_name/.openai_api_key
   ```

2. **プログラムの実行**：

   ```bash
   sudo bash system_checking_by_chatgpt.sh
   ```

   プログラムは次のように表示します：

   ```
   [INFO] プログラムの実行を開始します...
   [INFO] APIキーが正常に読み込まれました。
   [INFO] システム情報の収集を開始します...
   [INFO] システム情報の収集が完了しました。情報は /var/log/system_checking_by_chatgpt/raw_report_2024-12-12.txt に保存されました。
   [INFO] /var/log/system_checking_by_chatgpt/chunks_2024-12-12/chunk_aa の要約リクエストを行っています...
   ...
   [INFO] 最終レポートが生成されました: /var/log/system_checking_by_chatgpt/final_report_2024-12-12.md
   [INFO] プログラムの実行が完了しました。
   ```

3. **レポートの確認**：

   ```bash
   less /var/log/system_checking_by_chatgpt/final_report_2024-12-12.md
   ```

4. **デバッグログの確認（必要に応じて）**：
   ```bash
   less /var/log/system_checking_by_chatgpt/debug_2024-12-12.log
   ```

## プログラムコード

```shell title="system_checking_by_chatgpt.sh"
#!/usr/bin/env bash

set -euo pipefail
# set -x  # 必要に応じてデバッグ用に有効化

########################################
# 基本設定およびチェック
########################################

DATE=$(date +"%Y-%m-%d")
WORK_DIR="/var/log/system_checking_by_chatgpt"
RAW_REPORT="$WORK_DIR/raw_report_$DATE.txt"
CHUNKS_DIR="$WORK_DIR/chunks_$DATE"
SUMMARY_DIR="$WORK_DIR/summary_$DATE"
FINAL_REPORT="$WORK_DIR/final_report_$DATE.md"  # Markdown形式で出力
DEBUG_LOG="$WORK_DIR/debug_$DATE.log"

mkdir -p "$WORK_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p "$SUMMARY_DIR"

# 必須コマンドの存在チェック
REQUIRED_COMMANDS=("curl" "jq" "smartctl" "systemctl" "docker" "sensors")
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[ERROR] 必須コマンド $cmd がインストールされていません。" | tee -a "$DEBUG_LOG"
        exit 1
    fi
done

# ディスク空き容量のチェック（100MB以上必要）
AVAILABLE_SPACE=$(df "$WORK_DIR" | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 102400 ]; then
    echo "[ERROR] $WORK_DIR のディスク空き容量が不足しています。" | tee -a "$DEBUG_LOG"
    exit 1
fi

# ログ出力関数（標準エラー出力へ出力し、関数戻り値を妨げない）
log() {
    local LEVEL=$1
    shift
    local MESSAGE="$@"
    {
        echo "[$LEVEL] $MESSAGE" | tee -a "$DEBUG_LOG" >&2
        logger -t "system_checking_by_chatgpt" "[$LEVEL] $MESSAGE"
    }
}

log "INFO" "スクリプト実行開始..."

########################################
# APIキー読み込み
########################################

OPENAI_KEY_FILE="/home/your_user_name/.openai_api_key"
if [ ! -f "$OPENAI_KEY_FILE" ]; then
    log "ERROR" "$OPENAI_KEY_FILE が存在しません。"
    exit 1
fi

OPENAI_API_KEY=$(cat "$OPENAI_KEY_FILE")
if [ -z "$OPENAI_API_KEY" ]; then
    log "ERROR" "OPENAI_API_KEY が空です。"
    exit 1
fi
export OPENAI_API_KEY

log "INFO" "APIキーを正常に読み込みました。"

API_URL="https://api.openai.com/v1/chat/completions"
MODEL="gpt-4o"

########################################
# 関数定義
########################################

collect_system_info() {
    {
        echo "=== システム稼働時間 (uptime) ==="
        uptime
        echo ""

        echo "=== 日時情報 ==="
        date
        echo ""

        echo "=== CPU・メモリ使用状況 ==="
        top -b -n1 | head -n 20
        echo ""

        echo "=== メモリ使用状況 (free) ==="
        free -h
        echo ""

        echo "=== ディスク使用量 (df) ==="
        df -h
        echo ""

        echo "=== 異常なサービス一覧 (Failed Services) ==="
        systemctl list-units --state=failed
        echo ""

        echo "=== 温度センサー情報 ==="
        sensors 2>/dev/null || echo "[WARNING] 温度情報を取得できません。"
        echo ""

        echo "=== NVMe / SMART 状態 ==="
        sudo smartctl -a /dev/nvme0n1 2>/dev/null || echo "[WARNING] /dev/nvme0n1 をチェックできません。"
        echo ""
        sudo smartctl -a /dev/nvme1n1 2>/dev/null || echo "[WARNING] /dev/nvme1n1 をチェックできません。"
        echo ""

        echo "=== dmesg (直近100行) ==="
        dmesg | tail -n 100
        echo ""

        echo "=== システムジャーナル (エラーのみ直近300行) ==="
        journalctl -p err -n 300
        echo ""

        echo "=== ネットワーク状況 (ss -tulpn) ==="
        ss -tulpn 2>/dev/null || echo "[WARNING] ネットワーク状況を取得できません。"
        echo ""

        echo "=== ネットワークインターフェイス統計 (ip -s link) ==="
        ip -s link
        echo ""

        echo "=== Dockerコンテナ一覧 (ある場合) ==="
        docker ps -a 2>/dev/null || echo "[INFO] Dockerは起動していないか、インストールされていません。"
        echo ""

    } > "$RAW_REPORT"
}

generate_chunk_summaries() {
    # RAW_REPORTをチャンク分割
    CHUNK_SIZE=200
    split -l $CHUNK_SIZE "$RAW_REPORT" "$CHUNKS_DIR/chunk_"

    CHUNKS=("$CHUNKS_DIR"/chunk_*)

    if [ ${#CHUNKS[@]} -eq 0 ]; then
        log "ERROR" "チャンクが生成されませんでした。"
        exit 1
    fi

    # チャンクごとの要約用プロンプト
    CHUNK_SYSTEM_PROMPT="あなたはプロのシステム管理コンサルタントです。以下はシステム情報の一部です。この情報から以下を抽出して要約してください： (1) 重要なシステム状態情報（CPU/メモリ使用率、ディスク使用率、ネットワーク状況、サービス稼働状況、エラーメッセージなど）、(2) 異常やエラー、注意が必要な事象。可能な限り詳細に、かつ整理された形で記述してください。"

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
                    {"role": "system", "content": $system_prompt},
                    {"role": "user", "content": $user_prompt}
                ]
            }')

        log "INFO" "$chunk_file の要約リクエストを実行します..."
        START_TIME=$(date +%s)
        RESPONSE=$(curl -sS -X POST "$API_URL" \
          -H "Authorization: Bearer $OPENAI_API_KEY" \
          -H "Content-Type: application/json" \
          -d "$CHUNK_API_PAYLOAD")

        if [ $? -ne 0 ]; then
            log "ERROR" "ChatGPT APIとの通信に失敗 (要約段階): $chunk_file"
            exit 1
        fi
        END_TIME=$(date +%s)
        log "INFO" "APIリクエスト所要時間: $((END_TIME - START_TIME)) 秒。"

        SUMMARY=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>>"$DEBUG_LOG" || echo "")

        if [ -z "$SUMMARY" ] || [ "$SUMMARY" = "null" ]; then
            log "ERROR" "$chunk_file の要約取得に失敗しました。"
            echo "Raw Response: $RESPONSE" >> "$DEBUG_LOG"
            exit 1
        fi

        SUMMARY_FILE="$SUMMARY_DIR/summary_$(basename "$chunk_file").txt"
        echo "$SUMMARY" > "$SUMMARY_FILE"
        summaries+=("$SUMMARY_FILE")
    done

    # 要約ファイルリストを標準出力へ返す
    for s in "${summaries[@]}"; do
        echo "$s"
    done
}

combine_summaries() {
    local summaries=("$@")
    COMBINED_SUMMARY="$WORK_DIR/combined_summary_$DATE.txt"
    rm -f "$COMBINED_SUMMARY"
    touch "$COMBINED_SUMMARY"

    log "INFO" "全チャンク要約を統合します..."
    {
        echo "以下は複数チャンクの要約を統合した結果です："
        echo "------------------------------------"
        for sfile in "${summaries[@]}"; do
            echo "=== チャンク要約 ==="
            cat "$sfile"
            echo ""
        done
    } > "$COMBINED_SUMMARY"

    echo "$COMBINED_SUMMARY"
}

generate_final_report() {
    local COMBINED_SUMMARY="$1"

    # HEREDOCを使ってプロンプトを記述
    FINAL_SYSTEM_PROMPT=$(cat <<EOF
あなたはプロのシステム管理コンサルタントです。以下は複数のチャンクからまとめたシステム情報要約です。これに基づいて、**非常に詳細な最終報告書**を **Markdown形式** で作成してください。報告書には、明確なMarkdown見出しを付けた以下3つの主要セクションを含めてください。

**1. システム現況の記述**：
CPU、メモリ、ディスクなどのリソース使用状況、サービスの稼働状況、ネットワーク状態、重要なログ情報、エラーメッセージ、異常な事象を列挙・記述してください。

**2. システム現況の分析**：
上記情報を分析し、潜在的な問題点、異常原因の推測、パフォーマンスボトルネック、特異な状況やリスクを指摘してください。

**3. 改善策の提案**：
分析結果に基づき、実行可能で具体的な改善策、診断手順、パフォーマンス最適化、エラー修正、システム安定性を強化するための提案を示してください。

論理的で分かりやすく、十分な情報量を確保した詳細な報告書にしてください。
EOF
)

    FINAL_INPUT=$(cat "$COMBINED_SUMMARY")

    FINAL_API_PAYLOAD=$(jq -n \
        --arg system_prompt "$FINAL_SYSTEM_PROMPT" \
        --arg user_prompt "$FINAL_INPUT" \
        --arg model "$MODEL" \
        '{
            "model": $model,
            "messages": [
                {"role": "system", "content": $system_prompt},
                {"role": "user", "content": $user_prompt}
            ]
        }')

    log "INFO" "最終分析を開始します..."
    log "DEBUG" "最終分析リクエストペイロード: $FINAL_API_PAYLOAD"

    START_TIME=$(date +%s)
    RESPONSE=$(curl -sS -X POST "$API_URL" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d "$FINAL_API_PAYLOAD")

    if [ $? -ne 0 ]; then
        log "ERROR" "ChatGPT APIとの通信に失敗 (最終分析段階)"
        exit 1
    fi
    END_TIME=$(date +%s)
    log "INFO" "APIリクエスト所要時間: $((END_TIME - START_TIME)) 秒。"

    log "DEBUG" "最終分析レスポンス: $RESPONSE"

    FINAL_ANALYSIS=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>>"$DEBUG_LOG" || echo "")

    if [ -z "$FINAL_ANALYSIS" ] || [ "$FINAL_ANALYSIS" = "null" ]; then
        log "ERROR" "最終分析結果の取得に失敗しました。"
        echo "Raw Response: $RESPONSE" >> "$DEBUG_LOG"
        exit 1
    fi

    {
        echo "# 日次システム点検報告書 - $DATE"
        echo ""
        echo "$FINAL_ANALYSIS"
    } > "$FINAL_REPORT"

    log "INFO" "最終報告書を生成しました: $FINAL_REPORT"
}

########################################
# メイン処理フロー
########################################

log "INFO" "システム情報を収集します..."
collect_system_info
log "INFO" "システム情報を収集しました: $RAW_REPORT"

# generate_chunk_summariesの出力を配列summariesとして取得
mapfile -t summaries < <(generate_chunk_summaries)
COMBINED_SUMMARY_FILE=$(combine_summaries "${summaries[@]}")
generate_final_report "$COMBINED_SUMMARY_FILE"

log "INFO" "スクリプト実行完了。"
```

## 実行結果の表示

```markdown
# 毎日システムチェックレポート - 2024-12-12

## 詳細な最終報告

### 1. システムの現状説明

**CPU 使用状況**：

- CPU 使用率が高く、特に‘pt_data+’タスクの処理時に 200%〜300%を超えて使用しています。
- 複数の CPU コアの温度が 80°C を超えており、過熱の問題がある可能性があります。

**メモリ使用状況**：

- 合計メモリ 128577.3 MiB のうち、48421.6 MiB が使用中、76743.1 MiB がキャッシュ/バッファとして使用されています。
- システムでは「Out of memory」のエラーが頻発し、一部のプロセスはメモリ不足で強制終了されました。

**ディスク使用状況**：

- `/`（ルートディレクトリ）は 70%が使用中、/dev/nvme1n1 は 51%使用されています。
- NVMe ディスクの健康状態は「PASSED」ですが、一部「Invalid Field in Command」のエラーが表示されています。

**ネットワーク状態**：

- ネットワークカード`enp5s0`と`r8169`で「NETDEV WATCHDOG」の警告が多発しており、ネットワーク転送キューのタイムアウトが発生しています。
- ログには、192.168.0.1 からのマルチキャストパケットが UFW ファイアウォールでブロックされ続けている記録があります。
- 1 回の SSH 接続失敗が記録されています。

**サービスの実行状況**：

- `snap.firmware-updater.firmware-notifier`サービスが何度も起動に失敗しています。
- `NETDEV WATCHDOG`エラーログが多発しています。
- 一部のアプリケーション（`NetworkManager-dispatcher`サービスなど）が正常に起動できていません。

**セキュリティ状況**：

- ログには複数の`apparmor`拒否警告が表示されており、セキュリティ設定の確認が必要です。

### 2. システムの現状分析

**CPU 過熱と高使用率**：

- 高い CPU 使用率と温度は、高負荷の計算タスク（‘pt_data+’など）によるものと思われます。プロセスが必要かどうか、また計算の最適化が可能かを確認する必要があります。

**メモリボトルネック**：

- 「Out of memory」エラーが頻繁に発生しており、メモリの割り当てが不適切または不足していることが示唆されています。メモリ集中的なアプリケーションの状況をさらに確認する必要があります。

**ディスクの健康状態とエラー**：

- NVMe ディスクの「Invalid Field in Command」エラーは、ドライバーやファームウェアの問題が原因である可能性があります。更新が必要かどうかを確認する必要があります。

**ネットワークの問題**：

- 「NETDEV WATCHDOG」の警告は、ネットワークドライバや設定の問題を示唆しています。ネットワークカードのドライバや設定の更新を行う必要があります。
- 継続的な UFW ブロックイベントは、異常なトラフィックや設定ミスが原因である可能性があり、この IP デバイスを調べる必要があります。

**サービスの実行問題**：

- `snap.firmware-updater.firmware-notifier`サービスの問題は、リソースが不足しているか、依存するサービスの重大な故障が原因である可能性があります。

**セキュリティリスク**：

- UFW のブロックと`apparmor`の警告は潜在的なセキュリティリスクを示しており、防火壁やアプリケーションのセキュリティを確認する必要があります。

### 3. 解決策の提案

**CPU とメモリパフォーマンスの改善**：

1. **プロセスの確認と最適化**：

   - `top`または`htop`で CPU 使用率の高いプロセスを監視します。
   - 必要のない‘pt_data+’プロセスを停止または調整します。

2. **メモリ管理**：
   - `free -m`や`vmstat`コマンドを使用してメモリ状況を分析します。
   - 物理メモリの増設や swap 領域の調整を行います。

**ディスクエラーの修正**：

1. **ドライバ/ファームウェアの更新確認**：
   - `sudo smartctl -a /dev/nvme0n1`コマンドを実行して詳細な SMART 情報を確認します。
   - NVMe ドライバとファームウェアを更新します。

**ネットワーク問題の解決**：

1. **ネットワークカードのドライバ確認**：

   - `sudo ethtool -i r8169`でドライバのバージョンを確認し、更新が可能か調べます。
   - `journalctl -k | grep -i netdev`でネットワークカードのエラーログを詳細に調べます。

2. **UFW 設定の確認**：
   - `sudo ufw status verbose`コマンドを使用してアクティブなファイアウォールルールを確認します。
   - 192.168.0.1 からのトラフィックのソースを特定し、正当性を確認します。

**サービスとセキュリティの修正**：

1. **サービスの診断**：

   - `snap.service`関連のログを確認し、`journalctl -xe`でエラー情報を収集します。
   - `firmware-updater`の依存パッケージや設定を確認します。

2. **セキュリティ設定の強化**：
   - `cat /var/log/syslog | grep apparmor`で`apparmor`の警告情報を確認します。
   - 防火壁を強化し、ネットワークサービスのセキュリティポリシーを見直します。

最後に、システム全体の詳細なログ監視を行い、異常事象への迅速な対応能力を高め、システムの安定した運用を確保することをお勧めします。
```
