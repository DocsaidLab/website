---
slug: mac-selective-vpn-routing
title: VPN の選択的なトラフィックルーティングを設定する
authors: Zephyr
tags: [vpn-routing, macos]
image: /ja/img/2023/0901.webp
description: Mac 上で VPN ルーティングを設定する方法について解説します。
---

リモートワークで会社が用意した VPN を使用する場合でも、自宅の他のデバイスを利用したいことがあります。

<!-- truncate -->

## 設定の概要

今回は、Mac を使用して設定を行います。

## 問題の説明

例えば以下の状況を考えてみましょう：

- 会社の VPN ネットワーク：192.168.25.XXX
- 自宅のネットワーク：192.168.1.XXX

この場合、VPN を有効にするとすべてのトラフィックが会社のネットワークを経由します。その結果、自宅内の他のデバイスには接続できなくなります。

さらに、自宅で面白い動画を見ていると、会社のネットワーク管理者も一緒に笑っている…？(？？？)
何かがおかしい気がしますよね？

そこで、**VPN は会社のネットワークトラフィックのみを通し、それ以外のトラフィックはローカル経由にする**設定を行います。

:::tip
ここでは、VPN がすでに設定されており、正常に動作していることを前提としています。
もし VPN が正しく動作していない場合は、まず設定を確認してください。
:::

## 解決方法

### 1. 会社のネットワークを確認

まず、会社内部のネットワークセグメントを確認します。例えば：

`192.168.25.XXX`

### 2. 設定ファイルを編集

以下のコマンドで設定ファイルを開きます：

```bash
sudo vim /etc/ppp/ip-up
```

以下の内容を入力し、自分のネットワークセグメントに置き換えてください：

:::warning
以下の例では、VPN のネットワークセグメントが `192.168.25.XXX` と仮定しています。自分の環境に合わせて変更してください。
:::

```bash
#!/bin/sh
/sbin/route add -net 192.168.25.0/24 -interface ppp0
```

#### コマンドの簡単な説明

1. **/sbin/route**：`route` コマンドのパス。ルーティングテーブルを設定・表示するためのコマンドです。
2. **-net 192.168.25.0/24**：ネットワークルートを指定します。この例では、`192.168.25.0` から `192.168.25.255` の範囲を表します。
3. **-interface ppp0**：どのネットワークインターフェイスを使用するかを指定します。この場合は `ppp0`（Point-to-Point Protocol）です。

このコマンドにより、`192.168.25.0/24` のネットワークトラフィックは `ppp0` インターフェイスを経由するように設定されます。

---

設定を保存して終了したら、以下のコマンドでファイルに実行権限を付与します：

```bash
sudo chmod 755 /etc/ppp/ip-up
```

## 問題が解決しない場合

一部のデバイスでまだ接続できない場合は、以下の手順を試してください：

1. Mac のシステム設定を開き、ネットワーク設定を表示します。
2. 以下の図のように操作します：

<figure style={{"width": "80%"}}>
![vpn-setting](./img/vpn-setting.jpg)
</figure>

- **ステップ 1**：システム設定を開き、「ネットワーク」を選択します。
- **ステップ 2**：VPN 接続の右側にある「…」をクリックします。
- **ステップ 3**：「サービス順序を設定」を選択します。
- **ステップ 4**：VPN の順序を Wi-Fi の後ろに移動します。

---

多くの場合、VPN のサービス順序を一番上に設定すると、すべてのトラフィックが優先的に VPN を通るようになります。ここで VPN を下に移動することで、先ほどのルーティング設定が適用されます。

これで設定は完了です。他のトラフィックを VPN 経由にしたい場合は、`ip-up` ファイルに追記してください。

## 参考資料

1. [shalyf/vpn_route.md](https://gist.github.com/shalyf/d50b0bbf30a4b5020d2b84f4ae8eb4e0)
2. [How to selectively route network traffic through VPN on Mac OS X Leopard?](https://superuser.com/questions/4904/how-to-selectively-route-network-traffic-through-vpn-on-mac-os-x-leopard)