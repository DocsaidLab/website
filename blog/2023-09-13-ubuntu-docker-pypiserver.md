---
slug: ubuntu-docker-pypiserver
title: Ubuntu, Docker and PyPi Server
authors: TSE
tags: [ubuntu, ssh]
---

隨著 Python 社群的持續發展，許多開發者和團隊選擇建立自己的私有 Python 包索引伺服器，來儲存和管理自家的 Python 套件。這不僅提供了更好的版本控制，也確保了軟體包的安全性。

在本文中，我們用 Docker 來建立一個 PyPi Server，並且在 Ubuntu 上運行。

<!--truncate-->

我們假設您已經在 Ubuntu 上安裝了 Docker，並且已經熟悉了 Docker 的基本操作。

## 1. 拉取 pypiserver 的 Docker 映像

```bash
docker pull pypiserver/pypiserver:latest
```

## 2. 創建一個目錄來存儲 Python 包

不囉唆，我們直接在家目錄下創建一個目錄來存儲 Python 包。

```bash
mkdir ~/packages
```

## 3. 設定 htpasswd

htpasswd 是一種用於存儲用戶名和密碼（經常用於基本 HTTP 身份驗證）的文件格式。

pypiserver 會使用此文件來驗證試圖上傳或下載套件的用戶。這是一種簡單但有效的方式，可以增強 pypiserver 的安全性。

要創建一個 `.htpasswd` 文件，您需要 `apache2-utils` 套件：

```bash
sudo apt install apache2-utils
```

然後，使用以下命令創建一個新的 .htpasswd 文件：

```bash
htpasswd -c ~/.htpasswd [username]
```

它會提示您輸入 `username` 的密碼。輸入密碼後，`.htpasswd` 文件會在您的家目錄下創建。

這時，您就可以使用上面提到的 `docker run` 命令來運行 `pypiserver`，並且使用您剛剛建立的 `.htpasswd` 文件進行身份驗證。

## 4. 將 pypiserver 掛載為背景服務

要將 Docker 容器作為背景服務運行，我們可以使用 Docker Compose 和 Systemd。

### 4.1 安裝 Docker Compose

如果您還沒有安裝 Docker Compose，首先進行安裝，請參考：

- [**安裝 Docker Compose 的官方文件**](https://docs.docker.com/compose/install/)

要注意的是 Docker Compose 最近有較大規模的更新，很多使用方式跟之前不一樣，最明顯的就是原本使用 `docker-compose` 的指令，現在都改成 `docker compose` 了。

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

### 4.2 創建文件

找個地方創建 `docker-compose.yml`，並填入以下內容：

您也可以參考 `pypiserver` 提供的範本：[**docker-compose.yml**](https://github.com/pypiserver/pypiserver/blob/master/docker-compose.yml)

```yaml
version: '3.3'
services:
  pypiserver:
    image: pypiserver/pypiserver:latest
    volumes:
      - /home/[您的使用者名稱]/auth:/data/auth
      - /home/[您的使用者名稱]/packages:/data/packages
    command: run -P /data/auth/.htpasswd -a update,download,list /data/packages
    ports:
      - "8080:8080"
```

- 請將上述的 [您的使用者名稱] 替換為您的實際使用者名稱。
- 您可以在這裡修改外部 port 映射值，例如改成：”18080:8080″。

### 4.3 創建 Systemd 服務

建立一個設定檔案：

```bash
sudo vim /etc/systemd/system/pypiserver.service
```

添加以下內容：

```bash
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
- 請確保您的 Docker 路徑正確，您可以使用 `which docker` 來確認。
- 我們基於新版的 `docker compose` 指令，而不是用 `docker-compose`。

### 4.4 啟動 pypiserver 服務

告知 systemd 重新讀取新的服務設定：

```bash
sudo systemctl daemon-reload
```

啟動服務：

```bash
sudo systemctl start pypiserver.service
sudo systemctl enable pypiserver.service
```

這樣一來，`pypiserver` 就會作為一個 `systemd` 服務運行，並且每次主機啟動時它都會自動啟動。

## 5. 查看狀態

如果您想查看服務的當前狀態，您可以使用：

```bash
sudo systemctl status pypiserver.service
```

這將會顯示 `pypiserver` 服務的當前狀態，包括是否正在運行，以及最近的日誌輸出。

![pypiserver status](./resource/pypiserver.jpg)

## 6. 使用 pypiserver

現在，您可以使用 `pip` 來安裝和上傳套件了。

### 6.1 上傳套件

首先，您需要有一個已經打包的 Python 軟體包（通常是 .whl 或 .tar.gz 格式）。假設您已有一個名為 `example_package-0.1-py3-none-any.whl` 的包。

要上傳軟體包到您的 `pypiserver`，使用 `twine`：

```bash
pip install twine
twine upload --repository-url http://localhost:8080/ example_package-0.1-py3-none-any.whl
```

- 您需要確保 localhost:8080 是您的 pypiserver 服務的地址和端口。

### 6.2 安裝套件

要安裝套件，您需要使用 `pip`，並且指定您的 `pypiserver` 服務的地址和端口：

```bash
pip install --index-url http://localhost:8080/ example_package
```

### 6.3 使用基本認證

如果您的 pypiserver 設置了基本認證（可能會為了安全原因這麼做），您在上傳或下載時需要提供認證資訊：

- 上傳套件：

    ```bash
    twine upload --repository-url http://localhost:8080/ --username [username] --password [password] example_package-0.1-py3-none-any.whl
    ```

- 安裝套件：

    ```bash
    pip install --index-url http://[username]:[password]@localhost:8080/ example_package
    ```

### 6.4 為長期使用設定 pip.conf

如果您經常從此伺服器安裝套件，您可能不想每次都在 `pip install` 時指定 `--index-url`。此時，您可以在 `pip.conf` 中設定預設的包索引源。

首先，找到或創建 `pip.conf` 文件，以下是按優先級順序可能存在於您機器上的文件：

- 優先級 1: 站點級別的配置文件：

    - `/home/[您的使用者名稱]/.pyenv/versions/3.8.18/envs/main/pip.conf`

- 優先級 2: 使用者級別的配置文件：

    - `/home/[您的使用者名稱]/.pip/pip.conf`
    - `/home/[您的使用者名稱]/.config/pip/pip.conf`

- 優先級 3: 全局級別的配置文件：

    - `/etc/pip.conf`
    - `/etc/xdg/pip/pip.conf`

所以記得要釐清一下，當前的 python 環境會使用哪一份檔案，並找到該檔案後加入以下內容：

```bash
[global]
index-url = http://[提供服務的主機IP位置]:8080/
trusted-host = [提供服務的主機IP位置]
```

再次，請確保替換 `[提供服務的主機IP位置]:8080` 為您 `pypiserver` 的正確地址和端口。

此後，當您使用 `pip install [package_name]`，`pip` 會自動使用設定在 `pip.conf` 的伺服器地址作為套件源。

## 7. 結語

現在，您已經成功地設立了自己的私有 PyPI 伺服器，並學會了如何上傳和下載套件。

透過 `pypiserver`，您不僅能夠更有效地管理自己的 Python 套件，同時也能確保您的套件在一個受保護的環境中。希望本文能夠對您帶來實際的幫助。


