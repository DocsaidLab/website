---
slug: mac-forticlient-vpn-split-routing
title: FortiClient VPN の分割ルーティング設定とルートテスト
authors: Z. Yuan
tags: [vpn-routing, macos, forticlient]
image: /ja/img/2025/0802.jpg
description: FortiClient を会社の内部ネットワークのみに限定し、それ以外の通信はすべてローカルに戻す。
---

以前、VPN のルーティングを変更する方法を共有したことがあります。

- 前提：[**VPN に選択的なトラフィックルーティングを設定する**](/blog/mac-selective-vpn-routing)

<!-- truncate -->

当時使用していた VPN は L2TP 方式で、`/etc/ppp/ip-up`スクリプトを使ってルートを変更できました。

しかし、仕事の都合で新しい VPN ツールである**FortiClient**に切り替えました。

すると、従来の方法は使えなくなりました。

:::info
FortiClient は Fortinet 社が提供するセキュアエンドポイントソリューションで、主な用途は以下の通りです：

1. **VPN 接続ツール**
   FortiClient は SSL VPN と IPSec VPN に対応しており、ユーザーが安全にリモートから社内ネットワークにアクセスできます。FortiGate ファイアウォールのポリシーと連携し、企業レベルのネットワークセキュリティと接続管理を提供します。

2. **エンドポイント保護**
   VPN 機能に加え、マルウェア防御、ウェブフィルタリング、行動解析などの機能を備え、ユーザー端末のセキュリティを強化します。

3. **Fortinet エコシステムとの統合**
   FortiGate、FortiAnalyzer、EMS などの Fortinet 製品と連携し、統一されたセキュリティ監視・管理を実現します。

4. **クロスプラットフォーム対応**
   Windows、macOS、Linux、iOS、Android など多様なプラットフォームをサポートし、企業内の一括展開と管理が容易です。

簡単に言えば、FortiClient はリモート接続とセキュリティ保護を兼ね備えた企業向けクライアントツールであり、特に社内ネットワーク接続の厳密な管理が求められる組織に適しています。
:::

## なぜ無効になるのか？

FortiClient は macOS 上で `utunX` という仮想インターフェースを作成し、以下のような **デフォルトルート** を積極的にプッシュします。よくある組み合わせは：

```
0.0.0.0/1      → utunX
128.0.0.0/1    → utunX
default        → utunX
```

これら 3 つを合わせると「すべてのトラフィックが VPN を経由する」ことになります。また、接続するたびにローカル設定が上書きされるため、単に `/etc/ppp/ip-up` だけでは分割ルーティングの役割を果たせません。

では、どうすれば良いのか？💢

ただの VPN ツールなのに、全てのトラフィックを奪おうとするのか？

## つまり？

今回の私の目的はシンプルで、以下の条件だけ満たせば良い：

- **会社の内部ネットワーク**（例：`10.0.0.0/8`）は VPN 経由
- **ローカル LAN／一般的なインターネット** は物理ネットワークインターフェース（Wi-Fi や有線 LAN など）を直接通る
- 現在の VPN インターフェースとローカルゲートウェイを自動検出
- ルート設定が正しく反映されているか検証できる仕組みを提供

以下、まずは手動コマンドを説明し、最後に自動化スクリプトを紹介します。

---

## 1. ルートの修正

FortiClient は接続時に以下の 3 つの強制ルートを密かに追加し、すべてのトラフィックを VPN に引き込みます：

```
0.0.0.0/1
128.0.0.0/1
default
```

これら 3 つは IPv4 全体のネットワークをカバーしており、全てのネットワーク要求を丸ごと捕捉してしまいます。

なので最初にやることは、これらのルートを削除することです：

```bash
for cidr in default 0.0.0.0/1 128.0.0.0/1; do
  sudo route -q -n delete -ifscope "$VPN_IF" "$cidr"
done
```

> `$VPN_IF` は私のスクリプトで自動的にインターフェース（例：utun4、utun6 など）を見つけるための変数です。

このステップは非常に重要で、削除に失敗すると VPN が全トラフィックを奪い続けます。削除時に "not in table" と表示されることがありますが、それは該当ルートが存在しないだけなので無視して構いません。

---

## 2. ローカルのデフォルトルートを復元

VPN のデフォルトルートを削除したら、元々のローカルゲートウェイを戻す必要があります。

こうすることで `google.com` や `youtube.com` のような一般的なインターネット通信は物理ネットワークインターフェース（en0、en1 など）を経由します。

```bash
if ! sudo route -n change default "$LOCAL_GW"; then
  sudo route -n add default "$LOCAL_GW"
fi
```

> `$LOCAL_GW` は `netstat` コマンドなどから自動的に取得した元のゲートウェイ（例：192.168.0.1）で、手動入力は不要です。

---

## 3. 社内ネットワーク専用ルートを追加

次に VPN が社内ネットワークのトラフィックのみを扱うように設定します。

仮に社内ネットワークが `10.0.0.0/8` だとすると、VPN のピア先 IP にルートを向けます：

```bash
sudo route -n add -net 10.0.0.0/8 "$VPN_PEER"
```

既にルートが存在していれば `change` に切り替えます：

```bash
sudo route -n change -net 10.0.0.0/8 "$VPN_PEER"
```

> `$VPN_PEER` は VPN の Point-to-Point 対向先 IP で、スクリプト内で `ifconfig utunX` を解析して自動取得しています。

:::warning
もし社内ネットワークが 10.0.0.0/8 以外の場合は、この部分を必ず修正してください。例えば、社内ネットワークが 172.16.0.0/12 の場合は、以下のように対応させます：

```bash
sudo route -n add -net 172.16.0.0/12 "$VPN_PEER"
```

:::

---

## 4. 設定が有効か確認する

最後のステップは私が強化した部分です。

このスクリプトを実行し、いくつかのテストターゲットを指定すると、スクリプトは自動的に：

1. 各 IP またはホスト名が実際にどのインターフェースを通っているかをチェック
2. そのホストに対して ping を送り、通信可能かどうかを確認します

例えば：

スクリプトを実行し、テストターゲットとして以下を指定します：

```bash
bash forticlient_split.sh 10.1.1.1 192.168.0.100 google.com
```

すると、次のような出力が得られます：

```
🌐 10.1.1.1      ➜ インターフェース utun4 (期待値 utun4) ✅  接続成功 ✅
🌐 192.168.0.100 ➜ インターフェース en0   (期待値 ローカル) ✅  接続成功 ✅
🌐 google.com    ➜ インターフェース en0   (期待値 ローカル) ✅  接続成功 ✅
```

これにより「VPN は会社の内部ネットワークのみを担当する」という設定が正しく機能しているか、素早く検証できます。

接続状況やインターフェース、通信可否が一目でわかるので便利です！

なかなか良い感じでしょう？

---

## 完全版スクリプト

macOS、FortiClient VPN 対応、自動検出、エラーチェック、接続テスト、ルート表示、VPN ピア IP 解析などの機能を備えた完全版スクリプトを以下にまとめました。

このスクリプトを `forticlient_split.sh` という名前で保存し、実行権限を付与してから実行してください：

```bash
chmod +x forticlient_split.sh
./forticlient_split.sh 10.1.1.1 192.168.0.100 google.com
```

以下がコードになります。ご活用ください。

```bash title="forticlient_split.sh" showLineNumbers
#!/usr/bin/env bash
# =============================================================================
# FortiClient VPN 分割ルーティング（詳細出力付き）
#
# このスクリプトは FortiClient によって挿入されたルートをリセットし、
# VPN 以外のトラフィック用にローカルのデフォルトゲートウェイを復元し、
# 10.0.0.0/8 ネットワークを VPN にバインドし、
# ルーティングテーブル、インターフェースマッピング、接続テスト（ping）を詳細に表示します。
#
# 使い方:
#   forticlient_split.sh [ -h | --help ] host1 [host2 ...]
#
# オプション:
#   -h, --help    ヘルプメッセージを表示して終了します。
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# エラーと情報表示のヘルパー関数
# -----------------------------------------------------------------------------

# die はエラーメッセージを表示し、ステータス1で終了します。
# 引数:
#   $@: 表示するエラーメッセージ
die() {
  printf '❌ %s\n' "$*" >&2
  exit 1
}

# info は情報メッセージを表示します。
# 引数:
#   $@: 表示するメッセージ
info() {
  printf 'ℹ️  %s\n' "$*"
}

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------

# get_vpn_interface は最初に見つかった PtP アドレスを持つ utun/tun/ppp インターフェース名を出力します。
# 戻り値:
#   見つかったインターフェース名を出力、見つからなければ空出力
get_vpn_interface() {
  while read -r iface; do
    iface=${iface%:}
    if ifconfig "$iface" | grep -q 'inet .*-->'; then
      printf '%s\n' "$iface"
      return
    fi
  done < <(ifconfig | awk '/^(utun|tun|ppp)[0-9]+:/{print $1}')
}

# get_vpn_peer は指定された VPN インターフェースの PtP 対向 IP を取得します。
# 引数:
#   $1: インターフェース名（例: utun4）
# 戻り値:
#   対向 IP を出力、見つからなければ空出力
get_vpn_peer() {
  local iface="$1"
  ifconfig "$iface" | awk '/inet / && /-->/ {
    for (i = 1; i <= NF; i++) {
      if ($i == "-->") { print $(i+1); exit }
    }
  }'
}

# get_local_gateway は VPN 以外のデフォルトゲートウェイを取得します。
# 戻り値:
#   ゲートウェイ IP を出力、見つからなければ空出力
get_local_gateway() {
  netstat -rn \
    | awk '$1=="default" && $NF !~ /(utun|tun|ppp)/ { print $2; exit }'
}

# iface_of は指定ホストにアクセスする際に使われるインターフェース名を返します。
# 引数:
#   $1: ホスト名またはIP
# 戻り値:
#   インターフェース名を出力
iface_of() {
  route get "$1" 2>/dev/null \
    | awk '/interface:/{print $2}'
}

# show_route_info は現在のデフォルトルートと10.0.0.0/8のルート情報を表示します。
show_route_info() {
  printf '\n──────────────────────────────────────────────────\n'
  info "🔎 現在のデフォルトルート:"
  route get default 2>/dev/null | awk 'NR<=5'
  info "🔎 現在の 10.0.0.0/8 ルート:"
  route get 10.0.0.1 2>/dev/null | awk 'NR<=5'
  printf '──────────────────────────────────────────────────\n\n'
}

# test_connect は指定ホストへのping疎通確認を行います。
# 引数:
#   $1: ホスト名またはIP
test_connect() {
  local host="$1"
  printf '   ↳ %s への疎通テスト: ' "$host"
  if ping -c2 -W1 "$host" &>/dev/null; then
    printf '✅ 到達可能\n'
  else
    printf '❌ 到達不可\n'
  fi
}

# -----------------------------------------------------------------------------
# 使用方法表示
# -----------------------------------------------------------------------------

usage() {
  cat <<-EOF
使い方: $(basename "$0") [ -h | --help ] host1 [host2 ...]
オプション:
  -h, --help    ヘルプメッセージを表示して終了します。
EOF
  exit 0
}

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------

# フラグ解析
if [[ "${1-}" =~ ^-h|--help$ ]]; then
  usage
fi

# ホスト収集
HOSTS=("$@")
(( ${#HOSTS[@]} )) || die "少なくとも一つのホストを指定してください。"

# 1. VPNインターフェースとゲートウェイの検出
info "🔍 VPNインターフェースとローカルゲートウェイを検出中"
VPN_IF=$(get_vpn_interface)                         || die "VPNインターフェース（utun/tun/ppp）を検出できませんでした。"
VPN_PEER=$(get_vpn_peer "$VPN_IF")                  || die "VPNのPtP対向IPを取得できませんでした。"
LOCAL_GW=$(get_local_gateway)                       || die "ローカルゲートウェイを取得できませんでした。"
info "   • VPNインターフェース：$VPN_IF"
info "   • VPNピアIP：$VPN_PEER"
info "   • ローカルゲートウェイ：$LOCAL_GW"

# 変更前のルートを表示
show_route_info

# 2. FortiClient によって強制追加されたルートを削除
info "🧹 FortiClientが挿入した強制ルートを削除中"
for cidr in default 0.0.0.0/1 128.0.0.0/1; do
  sudo route -q -n delete -ifscope "$VPN_IF" "$cidr" 2>/dev/null || true
done

# 3. ローカルのデフォルトゲートウェイを復元
info "🚧 ローカルのデフォルトルートを $LOCAL_GW に設定中"
if ! sudo route -n change default "$LOCAL_GW" 2>/dev/null; then
  sudo route -n add default "$LOCAL_GW"
fi

# 4. 10.0.0.0/8 ネットワークをVPNにバインド
info "🔗 10.0.0.0/8 ネットワークを $VPN_PEER ($VPN_IF) にバインド"
if ! sudo route -n add -net 10.0.0.0/8 "$VPN_PEER" 2>/dev/null; then
  sudo route -n change -net 10.0.0.0/8 "$VPN_PEER"
fi

# 変更後のルートを表示
show_route_info

# 5. ルーティングインターフェースと接続性のテスト
info "🌐 ルートインターフェースと接続性をテスト中"
for host in "${HOSTS[@]}"; do
  actual_if=$(iface_of "$host" || echo "不明")
  if [[ $host == 10.* ]]; then
    expected_if="$VPN_IF"
  else
    expected_if="ローカル（非VPN）"
  fi
  printf '   • %-15s ➜ インターフェース %-8s (期待値 %s)\n' \
    "$host" "$actual_if" "$expected_if"
  test_connect "$host"
done

info "🎉 設定、ルート、接続性テストが完了しました"
```
