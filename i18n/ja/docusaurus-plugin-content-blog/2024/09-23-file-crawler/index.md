---
slug: file-crawler-python-implementation
title: ウェブページのファイルをダウンロードするPython実装
authors: Zephyr
image: /ja/img/2024/0923.webp
tags: [Python, File Crawler]
description: シンプルなウェブファイルダウンロードプログラム。
---

あるウェブページを見つけたところ、そこには数百個の PDF ファイルリンクが含まれていました。

エンジニアとして、一つずつ手動でクリックしてダウンロードするのは、ちょっと違いますよね？

そこで、小さなプログラムを書いて、すべてのファイルをダウンロードしてみましょう。

<!-- truncate -->

## パッケージのインストール

まず、必要なパッケージをインストールする必要があります。まだインストールしていない場合は、以下のコマンドを実行してください：

```bash
pip install requests beautifulsoup4 urllib3
```

## プログラムコード

さて、プログラムが完成したので、早速コードを共有します！

重要な箇所は枠で囲まれている部分で、あなたのニーズに応じて調整する必要があります。

```python {13,16} title="file_crawler.py"
import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ブラウザのヘッダーを模倣
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
}

# ウェブページのURL
url = "put_your_url_here"

# 対象ファイル形式
target_format = ".pdf"

# HTTP GETリクエストを送信し、ヘッダーを追加
response = requests.get(url, headers=headers)

# リクエストが成功したか確認
if response.status_code == 200:
    # BeautifulSoupでHTMLを解析
    soup = BeautifulSoup(response.text, "html.parser")

    # すべての<a>タグを検索し、href属性が対象形式に一致するリンクを抽出
    target_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(target_format): # ここでダウンロードするファイル形式を指定
            target_links.append(urljoin(url, href))

    # ファイルを保存するフォルダを作成
    os.makedirs("downloads", exist_ok=True)

    # 各ファイルをダウンロード
    for url in target_links:
        file_name = url.split("/")[-1]  # URLからファイル名を抽出
        file_path = os.path.join("downloads", file_name)

        # リクエストを送信してダウンロード
        response = requests.get(url, headers=headers)  # ヘッダーを同様に追加
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"ダウンロード完了: {file_name}")
        else:
            print(f"ダウンロード失敗: {url}")
else:
    print(f"ウェブページにアクセスできません。ステータスコード: {response.status_code}")
```

## プログラムの実行

コードが完成したら、プログラムを実行して、対象形式のすべてのファイルをダウンロードできます。

```bash
python file_crawler.py
```
