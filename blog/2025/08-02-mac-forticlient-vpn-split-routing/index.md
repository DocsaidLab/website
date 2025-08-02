---
slug: mac-forticlient-vpn-split-routing
title: FortiClient VPN 分流設定與路由測試
authors: Z. Yuan
tags: [vpn-routing, macos, forticlient]
image: /img/2025/0802.jpg
description: 讓 FortiClient 只負責公司內網，其他流量全部回到本地。
---

之前我有分享過怎麼改 VPN 路由。

- 前情提要：[**為 VPN 設定選擇性流量路由**](/blog/mac-selective-vpn-routing)

<!-- truncate -->

那時候用的 VPN 走的是 L2TP 的模式，可以透過 `/etc/ppp/ip-up` 腳本來修改路由。

後來我因為工作的需求，換了新的 VPN 工具：**FortiClient**。

然後原本的方法就失效了。

:::info
FortiClient 是 Fortinet 公司推出的一套安全端點解決方案，常見用途如下：

1. **VPN 連線工具**
   FortiClient 支援 SSL VPN 和 IPSec VPN，讓使用者能安全地從遠端存取公司內網。它可以整合 FortiGate 防火牆政策，提供企業級的網路安全與連線控管。

2. **端點防護**
   FortiClient 除了 VPN 功能，也可用來提供惡意程式防護、網頁過濾、行為分析等功能，強化使用者裝置的安全性。

3. **與 Fortinet 生態整合**
   它能與 Fortinet 的其他產品（例如 FortiGate、FortiAnalyzer、EMS）整合，提供統一的安全監控與管理機制。

4. **跨平台支援**
   FortiClient 支援 Windows、macOS、Linux、iOS、Android 等多平台，方便企業內部統一部署與管理。

簡單來說，FortiClient 是一套結合遠端連線與安全防護的企業用戶端工具，尤其適合需要嚴格控管內部網路連線權限的組織。
:::

## 為什麼失效？

FortiClient 會在 macOS 上建立 `utunX` 虛擬介面，並主動推送一整套 **預設路由**，常見組合有：

```
0.0.0.0/1      → utunX
128.0.0.0/1    → utunX
default        → utunX
```

這三條合起來等於「所有流量都經過 VPN」。而且每次連線時都會覆寫本地設定，所以單靠 `/etc/ppp/ip-up` 已經無法勝任分流的任務。

那怎麼行？💢

就只是一個 VPN 工具，就想拿走我全部的流量？

## 所以？

這次，我的目標很簡單，只要滿足以下條件：

- **公司內網**（例：`10.0.0.0/8`）走 VPN
- **本地區網／一般網際網路** 直接走實體網卡（Wi-Fi、乙太網路等）
- 自動偵測當前 VPN 介面與本地 gateway
- 提供驗證機制，確定路由真的生效

以下我們先講一下手動指令，自動化腳本放在最後面。

---

## 第一步：改路由

FortiClient 在連線時會偷偷塞進三條強制路由，把所有流量都拉進 VPN：

```
0.0.0.0/1
128.0.0.0/1
default
```

這三條加起來就是整個 IPv4 網段，也就是把你所有的網路請求一網打盡。

所以我們的第一步，就是把它們刪掉：

```bash
for cidr in default 0.0.0.0/1 128.0.0.0/1; do
  sudo route -q -n delete -ifscope "$VPN_IF" "$cidr"
done
```

> $VPN_IF 是我寫的腳本，目的是幫我們找出介面，例如 utun4、utun6 等。

這一步非常關鍵，如果沒有成功刪除，VPN 會持續攔截你的全部流量。刪除時也有可能出現 "not in table" 的訊息，只代表該路由不存在，可以略過。

---

## 第二步：恢復本地預設路由

VPN 的預設路由清掉之後，我們需要把原本的本地 gateway 補回來。

這樣才能讓像是 `google.com`、`youtube.com` 等一般網路流量回到實體網卡（en0、en1 等）。

```bash
if ! sudo route -n change default "$LOCAL_GW"; then
  sudo route -n add default "$LOCAL_GW"
fi
```

> `$LOCAL_GW` 是從 `netstat` 中自動取得的原本 gateway，例如 `192.168.0.1`，同樣不需手動輸入。

---

## 第三步：加入內網專屬路由

接下來要處理的是讓 VPN 只處理公司內部的流量。

假設內部網段是 `10.0.0.0/8`，我們會導到 VPN 的對端 IP：

```bash
sudo route -n add -net 10.0.0.0/8 "$VPN_PEER"
```

如果該路由已存在，則改用 `change`：

```bash
sudo route -n change -net 10.0.0.0/8 "$VPN_PEER"
```

> `$VPN_PEER` 是 VPN Point-to-Point 對端的 IP，由腳本透過 `ifconfig utunX` 分析自動取得。

:::warning
如果你的內網不是 10.0.0.0/8，記得要修改這段。例如公司內網是 172.16.0.0/12，就要對應調整成：

```bash
sudo route -n add -net 172.16.0.0/12 "$VPN_PEER"
```

:::

---

## 第四步：確認設定是否生效

最後這一步，是我加強的部分。

只要你執行這支腳本，並給幾個測試目標，程式就會自動：

1. 檢查每個 IP 或主機名實際走的是哪個介面
2. 嘗試對該主機送出 ping，看能否通

例如：

你只要執行腳本並輸入幾個測試目標，例如：

```bash
bash forticlient_split.sh 10.1.1.1 192.168.0.100 google.com
```

就會得到以下類似的輸出：

```
🌐 10.1.1.1      ➜ 介面 utun4 (期望 utun4) ✅  連通 ✅
🌐 192.168.0.100 ➜ 介面 en0   (期望 本地)   ✅  連通 ✅
🌐 google.com    ➜ 介面 en0   (期望 本地)   ✅  連通 ✅
```

這樣一來就能快速驗證你設定的「VPN 只負責公司內網」有沒有成功生效。

連線、介面、通訊一目了然！

看起來還不錯吧！

---

## 完整腳本

我將整套完整腳本整理在下方，支援 macOS、FortiClient VPN、自動偵測、錯誤提示、連通測試、路由印出、解析對端 IP 等功能。

你可以將這個腳本命名為 `forticlient_split.sh`，加上執行權限後直接執行：

```bash
chmod +x forticlient_split.sh
./forticlient_split.sh 10.1.1.1 192.168.0.100 google.com
```

以下為程式碼，祝你使用愉快：

```bash title="forticlient_split.sh" showLineNumbers
#!/usr/bin/env bash
# =============================================================================
# FortiClient VPN Split Routing with Enhanced Output
#
# This script resets FortiClient–injected routes, restores the local default
# gateway for non-VPN traffic, binds the 10.0.0.0/8 network to the VPN, and
# then shows rich diagnostics: routing tables, interface mappings, and
# connectivity tests (ping).
#
# Usage:
#   forticlient_split.sh [ -h | --help ] host1 [host2 ...]
#
# Options:
#   -h, --help    Show this help message and exit.
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Error and Info Helpers
# -----------------------------------------------------------------------------

# die prints an error message and exits with status 1.
# Arguments:
#   $@: The error message to display.
die() {
  printf '❌ %s\n' "$*" >&2
  exit 1
}

# info prints an informational message.
# Arguments:
#   $@: The message to display.
info() {
  printf 'ℹ️  %s\n' "$*"
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

# get_vpn_interface finds the first utun/tun/ppp interface with a PtP address.
# Returns:
#   Prints the interface name or nothing if not found.
get_vpn_interface() {
  while read -r iface; do
    iface=${iface%:}
    if ifconfig "$iface" | grep -q 'inet .*-->'; then
      printf '%s\n' "$iface"
      return
    fi
  done < <(ifconfig | awk '/^(utun|tun|ppp)[0-9]+:/{print $1}')
}

# get_vpn_peer obtains the Point-to-Point peer IP for a given VPN interface.
# Args:
#   $1: Interface name (e.g. utun4)
# Returns:
#   Prints the peer IP or nothing if not found.
get_vpn_peer() {
  local iface="$1"
  ifconfig "$iface" | awk '/inet / && /-->/ {
    for (i = 1; i <= NF; i++) {
      if ($i == "-->") { print $(i+1); exit }
    }
  }'
}

# get_local_gateway finds the non-VPN default gateway.
# Returns:
#   Prints the gateway IP or nothing if not found.
get_local_gateway() {
  netstat -rn \
    | awk '$1=="default" && $NF !~ /(utun|tun|ppp)/ { print $2; exit }'
}

# iface_of returns the interface used to reach a host.
# Args:
#   $1: Hostname or IP
# Returns:
#   Prints the interface name.
iface_of() {
  route get "$1" 2>/dev/null \
    | awk '/interface:/{print $2}'
}

# show_route_info prints current routes for default and 10.0.0.0/8.
show_route_info() {
  printf '\n──────────────────────────────────────────────────\n'
  info "🔎 當前 Default Route:"
  route get default 2>/dev/null | awk 'NR<=5'
  info "🔎 當前 10.0.0.0/8 Route:"
  route get 10.0.0.1 2>/dev/null | awk 'NR<=5'
  printf '──────────────────────────────────────────────────\n\n'
}

# test_connect tests ping connectivity to a host.
# Args:
#   $1: Hostname or IP
test_connect() {
  local host="$1"
  printf '   ↳ 測試 %s: ' "$host"
  if ping -c2 -W1 "$host" &>/dev/null; then
    printf '✅ 可達\n'
  else
    printf '❌ 不可達\n'
  fi
}

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------

usage() {
  cat <<-EOF
Usage: $(basename "$0") [ -h | --help ] host1 [host2 ...]
Options:
  -h, --help    Show this help message and exit.
EOF
  exit 0
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# Parse flags.
if [[ "${1-}" =~ ^-h|--help$ ]]; then
  usage
fi

# Collect hosts.
HOSTS=("$@")
(( ${#HOSTS[@]} )) || die "請指定至少一個 host。"

# 1. Detect VPN interface and gateways.
info "🔍 偵測 VPN 介面與本地 gateway"
VPN_IF=$(get_vpn_interface)                         || die "找不到 VPN 介面 (utun/tun/ppp)。"
VPN_PEER=$(get_vpn_peer "$VPN_IF")                  || die "無法取得 VPN PtP 對端 IP。"
LOCAL_GW=$(get_local_gateway)                       || die "無法取得本地 gateway。"
info "   • VPN 介面：$VPN_IF"
info "   • VPN Peer：$VPN_PEER"
info "   • 本地 Gateway：$LOCAL_GW"

# Show current routes before changes
show_route_info

# 2. Reset routes injected by FortiClient.
info "🧹 移除 FortiClient 強制注入的路由"
for cidr in default 0.0.0.0/1 128.0.0.0/1; do
  sudo route -q -n delete -ifscope "$VPN_IF" "$cidr" 2>/dev/null || true
done

# 3. Restore local default gateway.
info "🚧 設定本地 default ➜ $LOCAL_GW"
if ! sudo route -n change default "$LOCAL_GW" 2>/dev/null; then
  sudo route -n add default "$LOCAL_GW"
fi

# 4. Bind 10.0.0.0/8 to VPN.
info "🔗 綁定 10.0.0.0/8 ➜ $VPN_PEER ($VPN_IF)"
if ! sudo route -n add -net 10.0.0.0/8 "$VPN_PEER" 2>/dev/null; then
  sudo route -n change -net 10.0.0.0/8 "$VPN_PEER"
fi

# Show current routes after changes
show_route_info

# 5. Verify routing interfaces and connectivity.
info "🌐 測試路由介面與可達性"
for host in "${HOSTS[@]}"; do
  actual_if=$(iface_of "$host" || echo "未知")
  if [[ $host == 10.* ]]; then
    expected_if="$VPN_IF"
  else
    expected_if="本地 (非 VPN)"
  fi
  printf '   • %-15s ➜ 介面 %-8s (期望 %s)\n' \
    "$host" "$actual_if" "$expected_if"
  test_connect "$host"
done

info "🎉 完成設定、路由與連通性測試"
```
