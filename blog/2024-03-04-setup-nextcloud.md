---
slug: setup-nextcloud
title: 一起來建個私有雲吧！
authors: Zephyr
tags: [Nextcloud, Docker, Ubuntu]
---

之前我都把檔案放在 Google Drive 上，下載檔案的時候會透過 wget 指令來下載。

直到某一天，原本的下載指令突然就不能用了...

<!--truncate-->

既然如此，那我們來試試看 Nextcloud。

以下我們基於 Ubuntu 22.04 來進行相關配置。在開始之前，請你準備好一個域名，並且把這個域名指向你的伺服器。如果不知道該怎麼做，請直接 Google 搜尋『namecheap 怎麼用』。

## 安裝 Nextcloud

第一問：為什麼要用 Nextcloud？

- 我想要一個私有雲，不想要把檔案放在別人的伺服器上。

第二問：Nextcloud 跟 Owncloud 有什麼不一樣？

- Nextcloud 是由 Owncloud 的開發者分家出來的，兩者的功能差不多，但 Nextcloud 的開發速度比較快。

第三問：Nextcloud 要怎麼安裝？

- 這個問題比較複雜，因為 Nextcloud 的安裝方式有很多種，而且每一種都有不同的優缺點。
- 在本篇文章中，我唯一推薦的安裝方式是使用 Docker。

## 設定 Nextcloud All-in-One

- 參考官方文件：[**Nextcloud All-in-One**](https://github.com/nextcloud/all-in-one)

首先確保你已經安裝了 Docker 和 Docker Compose。如果還沒有完成，請 Google 搜尋 『Docker & Docker Compose 安裝方法』。

接著，建立一個 NextCloud 資料夾，然後寫一個 Docker Compose 的設定檔 `docker-compose.yml`：

```bash
mkdir nextcloud
vim nextcloud/docker-compose.yml
```

把以下的內容貼到 `docker-compose.yml` 裡面：

```yaml
services:
  nextcloud-aio-mastercontainer:
    image: nextcloud/all-in-one:latest
    init: true
    restart: always
    container_name: nextcloud-aio-mastercontainer
    volumes:
      - nextcloud_aio_mastercontainer:/mnt/docker-aio-config
      - /var/run/docker.sock:/var/run/docker.sock:ro
    ports:
      - 80:80
      - 8080:8080
      - 8443:8443
volumes:
  nextcloud_aio_mastercontainer:
    name: nextcloud_aio_mastercontainer
```

簡要說明一下上面的指令內容：

- `--init`：選項確保永遠不會建立殭屍行程。
- `--name nextcloud-aio-mastercontainer`：設定容器的名稱，這個名稱不允許更改，因為更改後可能會導致 mastercontainer 更新失敗。
- `--restart always`：設定容器的重啟策略為始終隨 Docker 守護程序一起啟動。
- `--publish 80:80`：將容器的80端口發佈到宿主機的80端口，用於獲取 AIO 接口的有效證書，如果不需要可以移除。
- `--publish 8080:8080`：將容器的 8080 端口發佈到宿主機的 8080 端口，此端口用於 AIO 接口，預設使用自簽名證書。 如果8080端口已被佔用，可以更改宿主機的端口，如：`--publish 8081:8080`。
- `--publish 8443:8443`：將容器的 8443 端口發佈到宿主機的8443端口，如果公開到網路，可以通過此端口訪問 AIO 接口並獲取有效證書，如果不需要可以移除。
- `--volume nextcloud_aio_mastercontainer:/mnt/docker-aio-config`：設定 mastercontainer 建立的檔案將儲存在名為 nextcloud_aio_mastercontainer 的 docker 磁碟區中，此設定不允許變更。
- `--volume /var/run/docker.sock:/var/run/docker.sock:ro`：將 docker 套接字掛載到容器內，用於啟動所有其他容器和其他功能。 在 Windows/macOS 和 docker 無根模式下需要調整。
- `nextcloud/all-in-one:latest`：指定使用的 Docker 容器映像。

還有更多詳細的設定，請參考官方文件：[compose.yaml](https://github.com/nextcloud/all-in-one/blob/main/compose.yaml)

## 設定系統服務

完成了上述的設定之後，再來是設定系統服務。

```bash
sudo vim /etc/systemd/system/nexcloud.service
```

貼上以下的內容：

```bash
[Unit]
Description=NextCloud Docker Compose
Requires=docker.service
After=docker.service

[Service]
WorkingDirectory=/home/[YourName]/nextcloud
ExecStart=/usr/bin/docker compose up --remove-orphans
ExecStop=/usr/bin/docker compose down
Restart=always

[Install]
WantedBy=multi-user.target
```

## 啟動 Nextcloud

```bash
sudo systemctl enable nextcloud
sudo systemctl start nextcloud
```

## 設定 Nextcloud

1. **訪問 Nextcloud AIO 介面**：
   - 在初始啟動後，你可以通過訪問 `https://ip.address.of.this.server:8080` 來開啟 Nextcloud AIO 介面，其中 `ip.address.of.this.server` 應該被替換為部署 Nextcloud 服務的伺服器的 IP 地址。
   - 重要的是使用 IP 地址而非域名來訪問這個連接埠（8080），因為 HTTP Strict Transport Security（HSTS）可能會阻止使用域名訪問。HSTS 是一種 Web 安全政策機制，它要求網絡瀏覽器僅通過安全的 HTTPS 連接與網站建立連接。

2. **自簽名證書的使用**：
   - 訪問 8080 端口時，可能會使用自簽名證書來確保通信的安全性。自簽名證書不是由受信任的證書機構（CA）發行的，因此瀏覽器可能會警告這個證書不可信。你需要在瀏覽器中手動接受這個證書，以便繼續訪問。

3. **獲取有效證書的自動化方式**：
   - 如果你的防火牆或路由器已開放或轉發了 80和 8443 端口，並且你已經將一個域名指向你的伺服器，那麼你可以通過訪問`https://your-domain-that-points-to-this-server.tld:8443`來自動獲取一個由受信任CA發行的有效證書，以增加安全性和便利性。這裡的`your-domain-that-points-to-this-server.tld`應該替換為你指向伺服器的實際域名。

4. **Nextcloud Talk 的連接埠開放**：
   - 為了確保 Nextcloud Talk 功能（如視訊通話和訊息）能夠正常運作，特別提到了需要在防火牆或路由器中為Talk 容器開放 3478/TCP 和 3478/UDP 連接埠。
   - 這些連接埠對於處理NAT穿越至關重要，NAT穿越是一種技術，允許網絡內外的裝置直接連接，這對於實時通信應用（如視訊通話）是必需的。

- **常見問題：**

    - **家用網路是動態 IP，怎麼指向域名？**

        - 我有試過一些 No-IP 的方法，後來覺得直接去中華電信申請固定 IP 最快最方便。

    - **我不想用 Docker，有沒有其他方法？**

        - 有，你可以直接安裝 Nextcloud，你需要自己來處理所有的依賴問題。
        - 相信我，很多坑。

    - **為什麼架好了卻連不上？**

        - 你的防火牆擋住了，如果防火牆沒開，那就是你的路由器擋住了。


---

輸入設定網址，你可以進入一個比後台還要後台的地方。

![login_1](./resource/login_1.jpg)

到這一步，你可能會驚恐地發現：『我沒有密碼啊！』

密碼會在你第一次登入的時候給你，但通常你會錯過他。這時候你可以透過以下的指令來找到密碼：

```bash
sudo grep password /var/lib/docker/volumes/nextcloud_aio_mastercontainer/_data/data/configuration.json
```

登入之後，你會看到一個設定畫面：

![login_2](./resource/login_2.jpg)

你看到的這個畫面已經是我設定完成的結果，如果你是第一次登入，在這裡，首先你要先輸入你之前準備好的網域，接著系統會要你再次下載一些 docker image，下載完成後，它會幫你啟動。

啟動之後，你就可以開始使用 Nextcloud 了。

## 結語

完成上面的步驟之後，在網址列中輸入你的網域後，你會看到一個很好看的介面，這個介面就是你的私有雲。

![login_3](./resource/login_3.jpg)

這個介面有很多功能，你可以透過這個介面來管理你的檔案，也可以透過這個介面來分享檔案。除此之外，你可以在手機上下載 Nextcloud 的 App，然後你可以直接透過手機來管理你的檔案。

有了 Nextcloud，你就不需要再擔心 Google Drive 的容量限制了。
