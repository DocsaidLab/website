---
slug: mac-forticlient-vpn-split-routing
title: FortiClient VPN Split Routing Configuration and Route Testing
authors: Z. Yuan
tags: [vpn-routing, macos, forticlient]
image: /en/img/2025/0802.jpg
description: Let FortiClient handle only the company intranet, while all other traffic returns locally.
---

I previously shared how to modify VPN routes.

- Background: [**Configuring Selective Traffic Routing for VPN**](/en/blog/mac-selective-vpn-routing)

<!-- truncate -->

Back then, the VPN used L2TP mode, and routes could be modified through the `/etc/ppp/ip-up` script.

Later, due to work requirements, I switched to a new VPN tool: **FortiClient**.

Then the original method stopped working.

:::info
FortiClient is a security endpoint solution developed by Fortinet, commonly used for the following purposes:

1. **VPN Connection Tool**
   FortiClient supports SSL VPN and IPSec VPN, enabling users to securely access the company intranet remotely. It can integrate with FortiGate firewall policies to provide enterprise-grade network security and connection control.

2. **Endpoint Protection**
   Besides VPN functionalities, FortiClient also offers malware protection, web filtering, behavior analysis, and other features to enhance the security of user devices.

3. **Integration with Fortinet Ecosystem**
   It integrates with other Fortinet products (such as FortiGate, FortiAnalyzer, EMS) to provide unified security monitoring and management.

4. **Cross-Platform Support**
   FortiClient supports multiple platforms including Windows, macOS, Linux, iOS, and Android, facilitating unified deployment and management within enterprises.

In short, FortiClient is an enterprise client tool combining remote connectivity and security protection, particularly suitable for organizations that require strict control over internal network access permissions.
:::

## Why did it fail?

FortiClient creates a `utunX` virtual interface on macOS and actively pushes a full set of **default routes**, typically including:

```
0.0.0.0/1      → utunX
128.0.0.0/1    → utunX
default        → utunX
```

These three routes combined mean "all traffic goes through the VPN." Also, every time it connects, it overwrites local settings, so relying on `/etc/ppp/ip-up` alone can no longer accomplish split routing.

What to do? 💢

It’s just a VPN tool, yet it wants to take all my traffic?

## So?

This time, my goal is simple, only to satisfy the following conditions:

- **Company intranet** (e.g., `10.0.0.0/8`) goes through the VPN
- **Local LAN/general internet** goes directly through the physical network interface (Wi-Fi, Ethernet, etc.)
- Automatically detect the current VPN interface and local gateway
- Provide validation to ensure the routes really take effect

Below, we first talk about manual commands; the automated script is at the end.

---

## Step 1: Modify routes

FortiClient secretly inserts three forced routes upon connection that pull all traffic into the VPN:

```
0.0.0.0/1
128.0.0.0/1
default
```

These three combined cover the entire IPv4 address space, effectively capturing all your network requests.

So our first step is to delete them:

```bash
for cidr in default 0.0.0.0/1 128.0.0.0/1; do
  sudo route -q -n delete -ifscope "$VPN_IF" "$cidr"
done
```

> `$VPN_IF` is determined by my script to find the interface, such as utun4, utun6, etc.

This step is critical. If you fail to delete these, the VPN will continue intercepting all your traffic. You might see "not in table" messages during deletion; these just mean the route doesn't exist and can be ignored.

---

## Step 2: Restore local default route

After clearing the VPN default routes, we need to restore the original local gateway.

This allows normal internet traffic, like to `google.com` or `youtube.com`, to go through the physical network interfaces (en0, en1, etc.).

```bash
if ! sudo route -n change default "$LOCAL_GW"; then
  sudo route -n add default "$LOCAL_GW"
fi
```

> `$LOCAL_GW` is automatically retrieved from `netstat`, e.g., `192.168.0.1`, no manual input needed.

---

## Step 3: Add intranet-specific route

Next, we handle routing so that only company internal traffic goes through the VPN.

Assuming the intranet is `10.0.0.0/8`, we route it to the VPN’s peer IP:

```bash
sudo route -n add -net 10.0.0.0/8 "$VPN_PEER"
```

If the route already exists, use `change` instead:

```bash
sudo route -n change -net 10.0.0.0/8 "$VPN_PEER"
```

> `$VPN_PEER` is the VPN point-to-point peer IP, automatically obtained by the script analyzing `ifconfig utunX`.

:::warning
If your intranet is not `10.0.0.0/8`, remember to modify this part accordingly. For example, if your company intranet is `172.16.0.0/12`, adjust it to:

```bash
sudo route -n add -net 172.16.0.0/12 "$VPN_PEER"
```

:::

---

## Step 4: Verify if settings take effect

This final step is my enhancement.

Once you run this script and supply some test targets, it will automatically:

1. Check which interface each IP or hostname actually uses
2. Attempt to ping the target to test connectivity

For example:

Simply run the script with a few test targets, such as:

```bash
bash forticlient_split.sh 10.1.1.1 192.168.0.100 google.com
```

You will get output similar to:

```
🌐 10.1.1.1       → interface utun4 (expected utun4) ✅ reachable ✅
🌐 192.168.0.100  → interface en0   (expected local) ✅ reachable ✅
🌐 google.com     → interface en0   (expected local) ✅ reachable ✅
```

This way, you can quickly verify whether your setting of "VPN only handling the company intranet" has taken effect successfully.

Connection, interface, and communication status are all clear at a glance!

Looks pretty good, right?

---

## Complete script

I have consolidated the full script below, supporting macOS, FortiClient VPN, auto-detection, error prompts, connectivity tests, route printing, and peer IP parsing.

You can save this script as `forticlient_split.sh`, make it executable, and run it directly:

```bash
chmod +x forticlient_split.sh
./forticlient_split.sh 10.1.1.1 192.168.0.100 google.com
```

Here’s the complete code, enjoy using it:

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
  info "🔎 Current Default Route:"
  route get default 2>/dev/null | awk 'NR<=5'
  info "🔎 Current 10.0.0.0/8 Route:"
  route get 10.0.0.1 2>/dev/null | awk 'NR<=5'
  printf '──────────────────────────────────────────────────\n\n'
}

# test_connect tests ping connectivity to a host.
# Args:
#   $1: Hostname or IP
test_connect() {
  local host="$1"
  printf '   ↳ Testing %s: ' "$host"
  if ping -c2 -W1 "$host" &>/dev/null; then
    printf '✅ Reachable\n'
  else
    printf '❌ Unreachable\n'
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
(( ${#HOSTS[@]} )) || die "Please specify at least one host."

# 1. Detect VPN interface and gateways.
info "🔍 Detecting VPN interface and local gateway"
VPN_IF=$(get_vpn_interface)                         || die "Cannot find VPN interface (utun/tun/ppp)."
VPN_PEER=$(get_vpn_peer "$VPN_IF")                  || die "Unable to get VPN Point-to-Point peer IP."
LOCAL_GW=$(get_local_gateway)                       || die "Cannot find local gateway."
info "   • VPN Interface: $VPN_IF"
info "   • VPN Peer: $VPN_PEER"
info "   • Local Gateway: $LOCAL_GW"

# Show current routes before changes
show_route_info

# 2. Reset routes injected by FortiClient.
info "🧹 Removing FortiClient injected forced routes"
for cidr in default 0.0.0.0/1 128.0.0.0/1; do
  sudo route -q -n delete -ifscope "$VPN_IF" "$cidr" 2>/dev/null || true
done

# 3. Restore local default gateway.
info "🚧 Setting local default route ➜ $LOCAL_GW"
if ! sudo route -n change default "$LOCAL_GW" 2>/dev/null; then
  sudo route -n add default "$LOCAL_GW"
fi

# 4. Bind 10.0.0.0/8 to VPN.
info "🔗 Binding 10.0.0.0/8 ➜ $VPN_PEER ($VPN_IF)"
if ! sudo route -n add -net 10.0.0.0/8 "$VPN_PEER" 2>/dev/null; then
  sudo route -n change -net 10.0.0.0/8 "$VPN_PEER"
fi

# Show current routes after changes
show_route_info

# 5. Verify routing interfaces and connectivity.
info "🌐 Testing routing interfaces and connectivity"
for host in "${HOSTS[@]}"; do
  actual_if=$(iface_of "$host" || echo "Unknown")
  if [[ $host == 10.* ]]; then
    expected_if="$VPN_IF"
  else
    expected_if="Local (non-VPN)"
  fi
  printf '   • %-15s ➜ Interface %-8s (expected %s)\n' \
    "$host" "$actual_if" "$expected_if"
  test_connect "$host"
done

info "🎉 Configuration, routing, and connectivity tests completed"
```
