---
slug: mac-selective-vpn-routing
title: 為 VPN 設定選擇性流量路由
authors: Z. Yuan
tags: [vpn-routing, macos]
image: /img/2023/0901.webp
description: 在 Mac 上設定 VPN 路由。
---

遠端辦公使用公司配置的 VPN，但我們仍想要使用本地端的其他的機器。

<!-- truncate -->

## 配置說明

我們使用 Mac 作為作業系統。

## 問題描述

舉例來說：

- 公司 VPN 網段：192.168.25.XXX
- 家裡的網段：192.168.1.XXX

這時候如果打開 VPN，所有的流量都會走到公司的網段，所以你將無法連上家裡相同網域內的其他機器。

不僅如此，你在家裡刷了個好笑的影片看著，殊不知公司的網管也跟著你一起笑了起來。(？？？)

好像哪裡怪怪的是不是？

所以我們要做的事情是：**讓 VPN 只導入公司的網段，其他的流量都走本地端。**

:::tip
這裡我們假設你已經將 VPN 設定好，並可以正常使用，現在只是要解決分流的問題。

如果 VPN 還不能正常使用，請先確認 VPN 的設定是否正確。
:::

## 解決問題

第一步，先確認公司內部的網段是什麼，例如：

192.168.25.XXX

接著，讓我們開啟一份文件：

```bash
sudo vim /etc/ppp/ip-up
```

輸入以下內容，請記得要換上你的網段：

:::warning
請注意，這裡的範例都是假設 VPN 的網段是 192.168.25.XXX，請你依照實際情況修改。
:::

```bash
#!/bin/sh
/sbin/route add -net 192.168.25.0/24 -interface ppp0
```

簡單解釋一下上面指令的意思：

1. **/sbin/route**：這是 route 命令的路徑，route 命令用於在網路中設定和顯示路由表。
2. **-net 192.168.25.0/24**：參數指定這是一個網路路由，而不是主機路由。 192.168.25.0/24 是網路地址和子網掩碼，它代表 192.168.25.0 到 192.168.25.255 的 IP 地址範圍。
3. **-interface ppp0**：指定路由應通過哪個網路介面。在這個例子中，它是 ppp0（點對點協議介面 0）。

整個指令的功能是通過 ppp0 介面添加一個到 192.168.25.0/24 網路的路由。

當你的系統試圖訪問 192.168.25.0/24 網路中的任何 IP 地址時，它將通過 ppp0 介面路由流量。

---

完成設定後，存檔退出，給予檔案權限：

```bash
sudo chmod 755 /etc/ppp/ip-up
```

## 還是壞了？

到這邊，可能有一部分的的機器還是無法上網，所以我們接著看：

打開 MacOC 的系統設定，找到網路，如下圖：

<figure style={{"width": "80%"}}>
![vpn-setting](./img/vpn-setting.jpg)
</figure>

- 步驟１：打開系統設定，點選「網路」
- 步驟２：點選旁邊小點點
- 步驟３：點選「設定服務順序」
- 步驟４：把 VPN 的順序拖到 Wi-Fi 後面

---

很多人設定完 VPN 後，會將 VPN 的服務順序拉到最上層，表示將所有的流量優先導入 VPN 內，所以這邊我們要把 VPN 拉下來，如此一來，上面的網路設定才能生效。

到這邊就完成了，之後有其他流量需要走 VPN 的話，就把位置寫到 ip-up 文件內即可。

## 參考資料

1. [shalyf/vpn_route.md](https://gist.github.com/shalyf/d50b0bbf30a4b5020d2b84f4ae8eb4e0)
2. [How to selectively route network traffic through VPN on Mac OS X Leopard?](https://superuser.com/questions/4904/how-to-selectively-route-network-traffic-through-vpn-on-mac-os-x-leopard)
