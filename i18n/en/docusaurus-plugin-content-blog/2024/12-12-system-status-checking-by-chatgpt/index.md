---
slug: system-status-checking-by-chatgpt
title: Automating Ubuntu System Status Checks with ChatGPT
authors: Z. Yuan
image: /en/img/2024/1212.webp
tags: [System-Monitoring, Automation, OpenAI-Integration]
description: Automate system status checks with ChatGPT.
---

To analyze issues, we need to check the system's status, but we may not be proficient in reading system logs.

In such cases, we can use ChatGPT to automate basic system status checks.

<!-- truncate -->

## Key Features

First, we collect system status and performance metrics using built-in Linux commands such as `uptime`, `top`, `free`, `df`, `systemctl`, `sensors`, `smartctl`, `dmesg`, `journalctl`, `ss`, `ip`, `docker`, etc.

Next, we divide the raw report into several sections, and use the GPT model (specified as `gpt-4o`) backed by the OpenAI API to automatically generate summaries and compile the information.

Finally, we combine multiple summaries into one, generating a system status report in Markdown format, which includes:

- System status description
- Problem analysis
- Improvement and recommendation proposals

:::warning
The code in this article may require adjustments as the OpenAI API is updated. Please ensure that the API usage aligns with the code.
:::

## Prerequisites

1. **Required Packages and Commands**:

   - `curl`: Used for HTTP communication with the API
   - `jq`: Used to parse JSON formatted API responses
   - `smartctl`: Used to check disk SMART information
   - `systemctl`: Used to check service status and failures
   - `docker`: Used to check Docker container status (ignore errors if Docker is not in use)
   - `sensors`: Used to retrieve system temperature information (if unavailable, the program will display a Warning)

   Ensure that these commands are correctly installed and can run properly on the system.

2. **OpenAI API Key**:

   This program will use the OpenAI API to generate system report summaries and final analysis.

   Ensure you have a valid OpenAI API key and store it in the specified path:

   - Default path: `/home/your_user_name/.openai_api_key`

   ```bash
   echo "YOUR_OPENAI_API_KEY" > /home/your_user_name/.openai_api_key
   chmod 600 /home/your_user_name/.openai_api_key
   ```

   Make sure to replace `YOUR_OPENAI_API_KEY` with your actual API Key.

3. **Working Directory and Disk Space**:

   The default working directory is `/var/log/system_checking_by_chatgpt`. The program will create the following files and folders here:

   - `raw_report_YYYY-MM-DD.txt`: Raw data collection report
   - `chunks_YYYY-MM-DD/`: Files after being chunked
   - `summary_YYYY-MM-DD/`: Files containing GPT summaries of each chunk
   - `combined_summary_YYYY-MM-DD.txt`: Final summary file combining all chunk summaries
   - `final_report_YYYY-MM-DD.md`: Final analysis report in Markdown format
   - `debug_YYYY-MM-DD.log`: Debug log file

   Ensure there is enough space in this directory (at least 100MB). The program will exit automatically if space is insufficient.

4. **Execution Permissions**:

   Ensure the program has executable permissions:

   ```bash
   chmod +x system_checking_by_chatgpt.sh
   ```

## Execution Method

Run the program in the terminal to start generating the report.

The expected execution flow is as follows:

1. **Initialization and Checks**: The program will check for the presence of all necessary external commands and confirm that the `OPENAI_API_KEY` is not empty. If there are any issues, it will immediately interrupt and display an error.
2. **System Information Collection**: The program will collect system information using various commands and consolidate the output into `raw_report_YYYY-MM-DD.txt`.
3. **Chunking and Summarization**: The program will divide `raw_report_YYYY-MM-DD.txt` into several subfiles of approximately 200 lines (`chunk_*`), and each subfile will be processed by the OpenAI API to generate summary files (`summary_chunk_*`), highlighting the key points and issues from that chunk.
4. **Merging All Summaries and Final Analysis**: The program will merge all summaries into a single file `combined_summary_YYYY-MM-DD.txt`, and then call the OpenAI API to generate the "final report."
5. **Generating Final Report**: The final report will be saved as `final_report_YYYY-MM-DD.md`, which will include:
   - System status description
   - Problem and anomaly analysis
   - Suggested solutions and diagnostic recommendations (which may include tables and command examples)

## Notes

1. **API Key Security**: Please keep the `OPENAI_API_KEY` secure. It should not be exposed, and it is recommended to restrict file read permissions (e.g., set to 600).
2. **Smart Adjustment of the Collection Scope**: If you wish to expand or reduce the system information collected, you can add or remove commands in the `collect_system_info()` function.
3. **OpenAI API Model and Token Limits**: By default, the model `MODEL="gpt-4o"` is used. Ensure this model is available within your API permissions. If the summaries or final report are too long and cause the API to fail or time out, consider adjusting the `CHUNK_SIZE` or reducing the output content.
4. **Timezone and Date Format**: The program uses `date +"%Y-%m-%d"` to format the date, which can be modified as needed.

## Example Flow

Below is a complete example of the running process:

1. **Verify Environment and API Key**:

   ```bash
   echo "sk-abc123xxx..." > /home/your_user_name/.openai_api_key
   chmod 600 /home/your_user_name/.openai_api_key
   ```

2. **Run the Program**:

   ```bash
   sudo bash system_checking_by_chatgpt.sh
   ```

   The program will display:

   ```
   [INFO] Starting program...
   [INFO] API key successfully loaded.
   [INFO] Starting system information collection...
   [INFO] System information collection complete, saved at /var/log/system_checking_by_chatgpt/raw_report_2024-12-12.txt
   [INFO] Requesting summary for /var/log/system_checking_by_chatgpt/chunks_2024-12-12/chunk_aa...
   ...
   [INFO] Final report generated: /var/log/system_checking_by_chatgpt/final_report_2024-12-12.md
   [INFO] Program execution complete.
   ```

3. **View the Report**:

   ```bash
   less /var/log/system_checking_by_chatgpt/final_report_2024-12-12.md
   ```

4. **Review the Debug Log (if needed)**:
   ```bash
   less /var/log/system_checking_by_chatgpt/debug_2024-12-12.log
   ```

## Code

```shell title="system_checking_by_chatgpt.sh"
#!/usr/bin/env bash

set -euo pipefail
# set -x  # Enable for debugging if needed

########################################
# Basic Setup and Checks
########################################

DATE=$(date +"%Y-%m-%d")
WORK_DIR="/var/log/system_checking_by_chatgpt"
RAW_REPORT="$WORK_DIR/raw_report_$DATE.txt"
CHUNKS_DIR="$WORK_DIR/chunks_$DATE"
SUMMARY_DIR="$WORK_DIR/summary_$DATE"
FINAL_REPORT="$WORK_DIR/final_report_$DATE.md"  # Using Markdown format
DEBUG_LOG="$WORK_DIR/debug_$DATE.log"

mkdir -p "$WORK_DIR"
mkdir -p "$CHUNKS_DIR"
mkdir -p "$SUMMARY_DIR"

# Check required commands
REQUIRED_COMMANDS=("curl" "jq" "smartctl" "systemctl" "docker" "sensors")
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[ERROR] Required command $cmd is not installed." | tee -a "$DEBUG_LOG"
        exit 1
    fi
done

# Check available disk space
AVAILABLE_SPACE=$(df "$WORK_DIR" | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 102400 ]; then # less than 100MB
    echo "[ERROR] Insufficient disk space in $WORK_DIR." | tee -a "$DEBUG_LOG"
    exit 1
fi

# Logging function: output to stderr to avoid interfering with function returns
log() {
    local LEVEL=$1
    shift
    local MESSAGE="$@"
    {
        echo "[$LEVEL] $MESSAGE" | tee -a "$DEBUG_LOG" >&2
        logger -t "system_checking_by_chatgpt" "[$LEVEL] $MESSAGE"
    }
}

log "INFO" "Script execution started..."

########################################
# Load API Key
########################################

OPENAI_KEY_FILE="/home/your_user_name/.openai_api_key"
if [ ! -f "$OPENAI_KEY_FILE" ]; then
    log "ERROR" "$OPENAI_KEY_FILE does not exist."
    exit 1
fi

OPENAI_API_KEY=$(cat "$OPENAI_KEY_FILE")
if [ -z "$OPENAI_API_KEY" ]; then
    log "ERROR" "OPENAI_API_KEY is empty."
    exit 1
fi
export OPENAI_API_KEY

log "INFO" "API key successfully loaded."

API_URL="https://api.openai.com/v1/chat/completions"
MODEL="gpt-4o"

########################################
# Function Definitions
########################################

collect_system_info() {
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
        sensors 2>/dev/null || echo "[WARNING] Could not retrieve temperature information."
        echo ""

        echo "=== NVMe / SMART Status ==="
        sudo smartctl -a /dev/nvme0n1 2>/dev/null || echo "[WARNING] Could not check /dev/nvme0n1"
        echo ""
        sudo smartctl -a /dev/nvme1n1 2>/dev/null || echo "[WARNING] Could not check /dev/nvme1n1"
        echo ""

        echo "=== Recent dmesg Entries (Last 100 lines) ==="
        dmesg | tail -n 100
        echo ""

        echo "=== System Journal (Last 300 lines, errors only) ==="
        journalctl -p err -n 300
        echo ""

        echo "=== Network Status (ss -tulpn) ==="
        ss -tulpn 2>/dev/null || echo "[WARNING] Could not retrieve network status."
        echo ""

        echo "=== Network Interface Statistics (ip -s link) ==="
        ip -s link
        echo ""

        echo "=== Docker Containers (if any) ==="
        docker ps -a 2>/dev/null || echo "[INFO] Docker is not running or not installed."
        echo ""

    } > "$RAW_REPORT"
}

generate_chunk_summaries() {
    # Split RAW_REPORT into chunks
    CHUNK_SIZE=200
    split -l $CHUNK_SIZE "$RAW_REPORT" "$CHUNKS_DIR/chunk_"

    CHUNKS=("$CHUNKS_DIR"/chunk_*)

    if [ ${#CHUNKS[@]} -eq 0 ]; then
        log "ERROR" "No chunks generated!"
        exit 1
    fi

    # Prompt for chunk summaries
    CHUNK_SYSTEM_PROMPT="You are a professional system administration consultant. The following is a portion of the system information. Please extract and summarize: (1) Important system status info (e.g., CPU/memory usage, disk usage, network status, service states, error messages), and (2) any abnormalities, errors, or issues requiring attention. Be as detailed as possible, while remaining clear and organized."

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

        log "INFO" "Requesting summary for $chunk_file ..."
        START_TIME=$(date +%s)
        RESPONSE=$(curl -sS -X POST "$API_URL" \
          -H "Authorization: Bearer $OPENAI_API_KEY" \
          -H "Content-Type: application/json" \
          -d "$CHUNK_API_PAYLOAD")

        if [ $? -ne 0 ]; then
            log "ERROR" "Failed to communicate with ChatGPT API during summary phase: $chunk_file"
            exit 1
        fi
        END_TIME=$(date +%s)
        log "INFO" "API request took $((END_TIME - START_TIME)) seconds."

        SUMMARY=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>>"$DEBUG_LOG" || echo "")

        if [ -z "$SUMMARY" ] || [ "$SUMMARY" = "null" ]; then
            log "ERROR" "No summary returned for chunk: $chunk_file"
            echo "Raw Response: $RESPONSE" >> "$DEBUG_LOG"
            exit 1
        fi

        SUMMARY_FILE="$SUMMARY_DIR/summary_$(basename "$chunk_file").txt"
        echo "$SUMMARY" > "$SUMMARY_FILE"
        summaries+=("$SUMMARY_FILE")
    done

    # Print summary file paths to stdout only
    for s in "${summaries[@]}"; do
        echo "$s"
    done
}

combine_summaries() {
    local summaries=("$@")
    COMBINED_SUMMARY="$WORK_DIR/combined_summary_$DATE.txt"
    rm -f "$COMBINED_SUMMARY"
    touch "$COMBINED_SUMMARY"

    log "INFO" "Combining all chunk summaries..."
    {
        echo "Below are the combined summaries from multiple chunks:"
        echo "------------------------------------"
        for sfile in "${summaries[@]}"; do
            echo "=== Chunk Summary ==="
            cat "$sfile"
            echo ""
        done
    } > "$COMBINED_SUMMARY"

    echo "$COMBINED_SUMMARY"
}

generate_final_report() {
    local COMBINED_SUMMARY="$1"

    # Use a HEREDOC for the final prompt to avoid quote issues
    FINAL_SYSTEM_PROMPT=$(cat <<EOF
You are a professional system administration consultant. Below is a combined set of summaries from multiple chunks of system information. Please produce a **very detailed final report** in **Markdown format** that includes the following three main sections with clear markdown headings:

**1. Current System Status**:
Describe the current resource usage (CPU, memory, disk), service states, network status, critical logs, error messages, and any abnormal events.

**2. System Status Analysis**:
Analyze the above information, highlight potential issues, explain possible causes for anomalies, identify performance bottlenecks, unusual conditions, or risks.

**3. Recommended Solutions**:
Based on your analysis, provide feasible and specific improvement suggestions, diagnostic steps, performance optimizations, error remediation steps, and advice to enhance system stability.

Maintain clarity, logical structure, and abundant detail.
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

    log "INFO" "Starting final analysis..."
    log "DEBUG" "Final analysis payload: $FINAL_API_PAYLOAD"

    START_TIME=$(date +%s)
    RESPONSE=$(curl -sS -X POST "$API_URL" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d "$FINAL_API_PAYLOAD")

    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to communicate with ChatGPT API during final analysis."
        exit 1
    fi
    END_TIME=$(date +%s)
    log "INFO" "API request took $((END_TIME - START_TIME)) seconds."

    log "DEBUG" "Final analysis response: $RESPONSE"

    FINAL_ANALYSIS=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>>"$DEBUG_LOG" || echo "")

    if [ -z "$FINAL_ANALYSIS" ] || [ "$FINAL_ANALYSIS" = "null" ]; then
        log "ERROR" "Failed to retrieve the final analysis."
        echo "Raw Response: $RESPONSE" >> "$DEBUG_LOG"
        exit 1
    fi

    {
        echo "# Daily System Check Report - $DATE"
        echo ""
        echo "$FINAL_ANALYSIS"
    } > "$FINAL_REPORT"

    log "INFO" "Final report generated: $FINAL_REPORT"
}

########################################
# Main Flow
########################################

log "INFO" "Collecting system information..."
collect_system_info
log "INFO" "System information collected and saved in $RAW_REPORT"

mapfile -t summaries < <(generate_chunk_summaries)
COMBINED_SUMMARY_FILE=$(combine_summaries "${summaries[@]}")
generate_final_report "$COMBINED_SUMMARY_FILE"

log "INFO" "Script execution completed."
```

## Report Example

```markdown
# Daily System Check Report - 2024-12-12

# Detailed System Status Report

## 1. Current System Status

### Resource Usage

- **Uptime and Load Average**: The system has been operating for 3 hours and 44 minutes, with a high load average of 7.06, 7.05, 7.09, which may indicate CPU strain if there are fewer cores than these values suggest.
- **CPU Usage**: The overall CPU utilization is 15.5% for user processes, 7.0% for system processes, and 77.5% idle. However, there are four processes utilizing over 100% CPU each, indicating potential multi-threaded applications engaging multiple cores.
- **Memory Usage**: The system has a total memory of 125 GiB, with 10 GiB free and 48 GiB actively used. The buffer/cache accounts for 76 GiB, indicating adequate available memory. Swap usage is negligible at 0 out of 2 GiB, suggesting effective memory management.
- **Disk Usage**: The root partition is 70% full with 1.2 TB used of 1.8 TB. The /data partition is at 51% usage with 881 GB used.

### Thermal and Smart Monitoring

- **CPU Temperatures**: Multiple CPU cores (e.g., Core 20, Package id 0, Core 28, Core 12) are running critically hot at 96°C, 85°C, and 84°C, indicating potential overheating issues.
- **NVMe Health**: Despite passing the health assessment, 85 "Invalid Field in Command" errors exist in the error log. The NVMe operates at 53°C, which is safe.

### Service States

- No failed services are reported besides consistent issues with `snap.firmware-updater.firmware-notifier.service` and a `fwupd-refresh.service` failure. Multiple failures are also noted for `NetworkManager-dispatcher.service`.

### Network Status

- **Network Interface**: `enp5s0` has multiple `NETDEV WATCHDOG` timeouts, indicating network communication issues. The system has firewall blocks affecting multicast traffic.
- **Listening Services**: Active services include Nginx (ports 80 and 443), Docker proxies (ports 8000 and 18080), and SSHD on a high port (20712).
- **Traffic**: Interfaces `veth321800f` and `vethc3fefc25` have normal TX/RX operations, while `veth814eefd` shows no RX data, suggesting possible configuration issues.

### Critical Logs and Errors

- **Out of Memory Events**: Several OOM events resulted in killing critical processes which may impact system stability.
- **Other Errors**: An SSH error involving a "Connection reset by peer" and 85 occurrences of "Invalid Field in Command" in SMART logs.

## 2. System Status Analysis

### Potential Issues

1. **High CPU Load**: The system load average is high, suggesting significant processing activities, possibly nearing core limits.
2. **Overheating**: CPU temperatures exceeding high thresholds pose a risk of thermal throttling or hardware damage.
3. **Disk Errors**: SMART errors could indicate misconfigurations or impending hardware issues needing investigation.
4. **Network Interruptions**: The recurring `NETDEV WATCHDOG` timeouts and blocked multicast traffic can degrade network performance.
5. **Memory Constraints**: Consistent OOM events denote memory overutilization or leaks, necessitating review of processes like `pipewire`, `xdg-permission-`, and more.

### Unusual Conditions or Risks

- **Failed Service Starts**: Persistent service failures around the firmware updater and NetworkManager services reveal potential system configuration or dependency issues.
- **Network Driver**: `r8169` driver might have compatibility problems with hardware leading to frequent timeouts and errors.
- **Security Alerts**: AppArmor and firewall logs indicate possible misconfigurations or unneeded traffic blocks that might impact network performance.

## 3. Recommended Solutions

### Immediate Actions:

- **Address CPU Overheating**: Evaluate cooling solutions, including checking fans, conducting thermal paste reapplications, or employing better case ventilation.
- **Investigate NVMe Errors**: Review firmware compatibility and update NVMe driver/firmware. Verify command syntaxes used by SMART tools.
- **Resolve OOM Events**: Analyze high-memory processes for inefficiencies or leaks, scale up physical memory, or fine-tune swapping parameters.

### Network Solutions:

- **Optimize NIC Configuration**: Update or reinstall the `r8169` driver, inspect physical network connectivity, and monitor for more accurate diagnostics on `enp5s0`.
- **Adjust Firewall Rules**: Review and possibly amend UFW rules to correctly handle multicast traffic without unnecessary blocks.

### Enhancing System Stability:

- **Review Service and Kernel Configurations**: Correct `snap.firmware-updater.firmware-notifier.service` and `NetworkManager-dispatcher.service` by checking service dependencies and configuration errors.
- **Implement Effective Monitoring**: Employ an extensive monitoring setup for regular checks on temperature, network performance, and memory.
- **System and Network Upgrades**: Should persistent high load and overheating be non-resolvable through tuning, consider hardware scaling, particularly for CPU and memory enhancements.
```
