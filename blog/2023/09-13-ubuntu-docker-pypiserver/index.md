---
slug: setting-up-pypiserver-on-ubuntu-with-docker
title: 搭建 PyPiServer 記錄
authors: Z. Yuan
tags: [docker, pypiserver]
image: /img/2023/0913.webp
description: 記錄 Ubuntu 上搭建 PyPiServer 的過程。
---

今天我們想用 Docker 來建立一個 PyPi Server，並且在 Ubuntu 上運行，在此紀錄一下過程。

我們假設你已經在 Ubuntu 上安裝了 Docker，並且已經熟悉了 Docker 的基本操作。

<!-- truncate -->

## 拉取映像

```bash
docker pull pypiserver/pypiserver:latest
```

## 建立目錄

不囉唆，我們直接在家目錄下建立一個目錄來存放 Python 包。

```bash
mkdir ~/packages
```

你可以換成你喜歡的名字，但是後面設定檔案也要跟著改。

## 設定 htpasswd

:::tip
如果你不想設定密碼，可以跳過這一步。
:::

htpasswd 是一種用於存儲用戶名和密碼的文件格式，pypiserver 會使用此文件來驗證使用者。

這是個簡單有效，又可以增強 pypiserver 安全性的方式。

我們先安裝一下 apache2-utils：

```bash
sudo apt install apache2-utils
```

然後，使用以下命令建立一個新的 `.htpasswd` 文件：

```bash
htpasswd -c ~/.htpasswd [username]
```

它會提示你輸入 `username` 的密碼。輸入密碼後，`.htpasswd` 文件會在家目錄下建立檔案。

建立檔案後，我們就可以使用上面提到的 `docker run` 命令來運行 `pypiserver`，並且使用 `.htpasswd` 文件驗證。

## 掛載為背景服務

要將 Docker 容器作為背景服務運行，我們在這裡使用 Docker Compose 和 Systemd。

### 安裝 Docker Compose

如果你還沒有安裝 Docker Compose，首先進行安裝，請參考：

- [**安裝 Docker Compose 的官方文件**](https://docs.docker.com/compose/install/)

要注意的是 Docker Compose 最近有較大規模的更新，很多使用方式跟之前不一樣。

最明顯的就是原本使用 `docker-compose` 的指令，現在一律改成 `docker compose` 。

我們根據官方文件，把安裝內容寫在下面：

安裝最新版本的 Docker Compose：

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

檢查 Docker Compose 是否正確安裝：

```bash
docker compose version
```

### 建立配置文件

找個地方創建 `docker-compose.yml`，並填入以下內容：

```yaml {6-7}
version: "3.3"
services:
  pypiserver:
    image: pypiserver/pypiserver:latest
    volumes:
      - /home/[使用者名稱]/auth:/data/auth
      - /home/[使用者名稱]/packages:/data/packages
    command: run -P /data/auth/.htpasswd -a update,download,list /data/packages
    ports:
      - "8080:8080"
```

- 請將上面重點劃記的 [使用者名稱] 替換為實際使用者名稱。
- 我們可以在這裡修改外部 port 映射值，例如改成："18080:8080"。

:::tip
可以參考 `pypiserver` 提供的範本：[**docker-compose.yml**](https://github.com/pypiserver/pypiserver/blob/master/docker-compose.yml)
:::

如果你在配置時不想設定密碼，請修改上面的 `command` 中的指令為：

```yaml
command: run -a . -P . /data/packages --server wsgiref
```

### 建立 Systemd 服務

建立一個設定檔案：

```bash
sudo vim /etc/systemd/system/pypiserver.service
```

寫入以下內容：

```bash {7}
[Unit]
Description=PypiServer Docker Compose
Requires=docker.service
After=docker.service

[Service]
WorkingDirectory=/path/to/your/docker-compose/directory
ExecStart=/usr/bin/docker compose up --remove-orphans
ExecStop=/usr/bin/docker compose down
Restart=always

[Install]
WantedBy=multi-user.target
```

- 請確保將 `/path/to/your/docker-compose/directory` 替換為 `docker-compose.yml` 文件的實際路徑，寫到路徑就可以，不用檔案名稱。
- 請確保 Docker 路徑正確，這裡可以使用 `which docker` 來確認。
- 我們基於新版的 `docker compose` 指令，而不是用 `docker-compose`。

### 啟動服務

告知 systemd 重新讀取新的服務設定：

```bash
sudo systemctl daemon-reload
```

啟動服務：

```bash
sudo systemctl enable pypiserver.service
sudo systemctl start pypiserver.service
```

## 查看狀態

如果你想查看服務的當前狀態，你可以使用：

```bash
sudo systemctl status pypiserver.service
```

這會顯示 `pypiserver` 服務的當前狀態，包括是否正在運行，以及最近的日誌輸出。

<div align="center">
<figure style={{"width": "80%"}}>
![pypiserver status](./img/pypiserver.jpg)
</figure>
</div>

## 開始使用

現在，我們可以使用 `pip` 來安裝和上傳套件了。

### 上傳套件

首先，假設我們已有一個名為 `example_package-0.1-py3-none-any.whl` 的包。

接著用 `twine` 這個工具來上傳套件：

```bash
pip install twine
twine upload --repository-url http://localhost:8080/ example_package-0.1-py3-none-any.whl
```

- 需要確保 localhost:8080 是你的 pypiserver 服務的地址和端口。

### 下載安裝套件

我們使用 `pip` 來安裝套件，安裝時，需要指定 `pypiserver` 服務的地址和端口：

```bash
pip install --index-url http://localhost:8080/ example_package
```

### 使用基本認證

如果你的 pypiserver 設置了基本認證，在上傳或下載時就需要你提供認證資訊：

- 上傳套件：

  ```bash
  twine upload \
    --repository-url http://localhost:8080/ \
    --username [username] \
    --password [password] \
    example_package-0.1-py3-none-any.whl
  ```

- 安裝套件：

  ```bash
  pip install \
    --index-url http://[username]:[password]@localhost:8080/ \
    example_package
  ```

## 設定 `pip.conf`

由於我們經常從此伺服器安裝套件，所以不想每次都在 `pip install` 時指定 `--index-url`。

因此我們可以把相關配置資訊寫在 `pip.conf` 內。

### 配置文件

`pip.conf` 文件可以存在很多個地方，我們可以按照優先級順序去找：

- 優先級 1: 站點級別的配置文件：

  - `/home/[使用者名稱]/.pyenv/versions/3.8.18/envs/main/pip.conf`

- 優先級 2: 使用者級別的配置文件：

  - `/home/[使用者名稱]/.pip/pip.conf`
  - `/home/[使用者名稱]/.config/pip/pip.conf`

- 優先級 3: 全局級別的配置文件：

  - `/etc/pip.conf`
  - `/etc/xdg/pip/pip.conf`

如果你也想設定，要先釐清一下當前的 python 環境會使用哪一份檔案，並找到該檔案後加入以下內容：

```bash
[global]
index-url = http://[提供服務的主機IP位置]:8080/
trusted-host = [提供服務的主機IP位置]
```

再次，請確保替換 `[提供服務的主機IP位置]:8080` 為你 `pypiserver` 的正確地址和端口。

設定完成後，當我們使用 `pip install [package_name]` 時，系統會自動使用設定在 `pip.conf` 的伺服器地址作為套件源。

## 結語

現在，你已經跟著我們成功地建立了自己的 PyPI 伺服器，並學會了如何上傳和下載套件。

希望本篇文章有解決你的問題。
