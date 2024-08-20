---
slug: mac-selective-vpn-routing
title: Setting Up Selective Traffic Routing for VPN on Mac
authors: Zephyr
tags: [routing, vpn, macos]
image: /en/img/2023/0901.webp
description: Configuring VPN routing on Mac.
---

<figure>
![title](/img/2023/0901.webp)
<figcaption>Cover Image: Automatically generated after reading this article by GPT-4</figcaption>
</figure>

---

When working from home using a company-configured VPN, it's common to still need access to other local machines. For instance:

<!-- truncate -->

- Company VPN subnet: 192.168.25.XXX
- Home subnet: 192.168.1.XXX

When you open the VPN, all traffic typically routes through the company's subnet, preventing access to other devices on your home network.

Moreover, innocently watching a funny video at home might unexpectedly draw the attention of your company's network administrators.

That doesn't sound right, does it?

So, what we want to do is: **route VPN traffic only to the company subnet, while letting other traffic go through the local network.**

:::info
I assume you've already set up your VPN and it's working fine; we're just addressing the routing issue here.
:::

## Solving the Issue

First, identify the internal network of your company, for example:

192.168.25.XXX

Then, let's open a file:

```bash
sudo vim /etc/ppp/ip-up
```

Enter the following content, but remember to replace it with your subnet:

:::warning
Note that the examples here assume the VPN subnet is 192.168.25.XXX; adjust as necessary.
:::

```bash
#!/bin/sh
/sbin/route add -net 192.168.25.0/24 -interface ppp0
```

Here's a brief explanation of the above command:

1. **/sbin/route**

   This is the path to the route command, used to set up and display the routing table in a network.

2. **-net 192.168.25.0/24**

   This parameter specifies a network route rather than a host route. 192.168.25.0/24 is the network address and subnet mask, representing the range of IP addresses from 192.168.25.0 to 192.168.25.255.

3. **-interface ppp0**

   Specifies through which network interface the route should pass. In this example, it's ppp0 (Point-to-Point Protocol interface 0).

The entire command adds a route to the 192.168.25.0/24 network via the ppp0 interface. When your system tries to access any IP address in the 192.168.25.0/24 network, it will route the traffic through the ppp0 interface.

Next, save and exit the file, and give it permissions:

```bash
sudo chmod 755 /etc/ppp/ip-up
```

## Still Not Working?

If some machines still can't access the internet, let's proceed with the following steps:

Open System Preferences on macOS, navigate to Network, as shown below:

- Step 1: Open System Preferences, click on "Network"
- Step 2: Click on the cog icon
- Step 3: Select "Set Service Order"
- Step 4: Drag the VPN below Wi-Fi

![vpn-setting](./img/vpn-setting.jpg)

The reason for this step is that after configuring the VPN, it may be set to the top of the service order, meaning all traffic is routed through the VPN. By moving the VPN down, the network settings above can take effect.

That's it! If there are other types of traffic that need to go through the VPN, simply add the appropriate routes to the ip-up file.

## References

1. [shalyf/vpn_route.md](https://gist.github.com/shalyf/d50b0bbf30a4b5020d2b84f4ae8eb4e0)
2. [How to selectively route network traffic through VPN on Mac OS X Leopard?](https://superuser.com/questions/4904/how-to-selectively-route-network-traffic-through-vpn-on-mac-os-x-leopard)
