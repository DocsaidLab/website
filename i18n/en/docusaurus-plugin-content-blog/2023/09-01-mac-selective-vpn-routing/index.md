---
slug: mac-selective-vpn-routing
title: Setting Up Selective Traffic Routing for VPN on Mac
authors: Zephyr
tags: [routing-vpn, macos]
image: /en/img/2023/0901.webp
description: Configuring VPN routing on Mac.
---

When working remotely with a company VPN setup, sometimes you still need access to local devices and resources on your home network.

<!-- truncate -->

## Configuration Guide

We’ll use Mac as the operating system for this setup.

## Problem Overview

For example:

- Company VPN network range: 192.168.25.XXX
- Home network range: 192.168.1.XXX

When the VPN is active, all traffic is routed through the company's network, preventing access to devices on your home network within the same domain.

In other words, even if you're at home streaming a funny video, the company's network admin might inadvertently catch a glimpse too. Something doesn’t feel quite right about this, does it?

Our goal here is to **route only the traffic meant for the company network through the VPN, while all other traffic stays on the local network.**

:::tip
This guide assumes that your VPN is already configured and functioning correctly. Here, we’ll only address the selective routing issue.

If the VPN isn’t working, please ensure your VPN settings are correct before proceeding.
:::

## Solving the Problem

### Step 1: Identify Your Company’s Internal Network Range

First, identify the network range for your company, for example:

192.168.25.XXX

Next, let’s open a system file to configure routing rules:

```bash
sudo vim /etc/ppp/ip-up
```

Add the following content, replacing the network range with your company’s network range:

:::warning
Note: This example assumes the VPN network range is 192.168.25.XXX. Modify according to your actual network setup.
:::

```bash
#!/bin/sh
/sbin/route add -net 192.168.25.0/24 -interface ppp0
```

Let’s break down what this command does:

1. **/sbin/route**: This is the path to the `route` command, which is used for configuring and displaying routing tables.
2. **-net 192.168.25.0/24**: This specifies that the route is a network route, not a host route. `192.168.25.0/24` represents the range of IP addresses from `192.168.25.0` to `192.168.25.255`.
3. **-interface ppp0**: This specifies the network interface for the route, in this case, `ppp0` (point-to-point protocol interface 0).

This command effectively adds a route through the `ppp0` interface for the `192.168.25.0/24` network range. When your system tries to access any IP address within this range, it will route the traffic through the `ppp0` interface.

---

After editing the file, save and exit, then give it the necessary permissions:

```bash
sudo chmod 755 /etc/ppp/ip-up
```

## Still Not Working?

At this point, some devices might still have trouble accessing the internet, so let’s try adjusting the network service order in macOS.

Open macOS’s System Preferences, and go to Network:

<figure style={{"width": "80%"}}>
![vpn-setting](./img/vpn-setting.jpg)
</figure>

- Step 1: Open System Preferences, and select “Network”
- Step 2: Click the small options button (three dots)
- Step 3: Select “Set Service Order”
- Step 4: Drag the VPN service below Wi-Fi in the order

---

Many people set the VPN at the top of the network service order, prioritizing all traffic through the VPN. Here, we’re lowering the VPN service order so that our custom network configuration can take effect.

That’s it! To route additional traffic through the VPN in the future, simply add the specific addresses in the `ip-up` file.

## References

1. [shalyf/vpn_route.md](https://gist.github.com/shalyf/d50b0bbf30a4b5020d2b84f4ae8eb4e0)
2. [How to selectively route network traffic through VPN on Mac OS X Leopard?](https://superuser.com/questions/4904/how-to-selectively-route-network-traffic-through-vpn-on-mac-os-x-leopard)
