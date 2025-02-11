---
slug: file-crawler-python-implementation
title: 下載網頁檔案的 Python 實作
authors: Z. Yuan
image: /img/2024/0923.webp
tags: [Python, File Crawler]
description: 一個簡單的網頁檔案下載程式。
---

我們看上了一個網頁，裡面有上百份 pdf 檔案連結。

身為工程師的我們，如果是自己逐篇點開下載，顯然不太對？

所以這裡就需要寫一個小程式，幫我們下載所有檔案。

<!-- truncate -->

## 安裝套件

首先，你需要安裝所需的套件，如果還沒安裝的話，可以透過以下命令來安裝：

```bash
pip install requests beautifulsoup4 urllib3
```

## 程式碼

話不多說，既然程式寫完了，我們就直接上程式碼吧！

重點框起來的部分，是你得自己修改的地方，請根據你的需求來調整程式碼。

```python {13,16} title="file_crawler.py"
import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# 模擬瀏覽器的 headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
}

# 網頁 URL
url = "put_your_url_here"

# 目標格式
target_format = ".pdf"

# 發送 HTTP GET 請求，添加 headers
response = requests.get(url, headers=headers)

# 檢查請求是否成功
if response.status_code == 200:
    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 查找所有的<a>標籤，篩選出 href 屬性符合目標格式的連結
    target_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(target_format): # 在這邊指定要下載的檔案格式
            target_links.append(urljoin(url, href))

    # 創建資料夾來儲存檔案
    os.makedirs("downloads", exist_ok=True)

    # 下載每個檔案
    for url in target_links:
        file_name = url.split("/")[-1]  # 從 URL 中提取檔名
        file_path = os.path.join("downloads", file_name)

        # 發送請求下載
        response = requests.get(url, headers=headers)  # 同樣添加 headers
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"已下載: {file_name}")
        else:
            print(f"無法下載: {url}")
else:
    print(f"無法存取網頁，狀態碼: {response.status_code}")
```

## 執行程式

完成後，可以直接執行程式，下載所有符合目標格式的檔案。

```bash
python file_crawler.py
```
