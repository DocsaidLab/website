---
slug: download-from-google-drive-using-python
title: 使用 Python 從 Google Drive 下載檔案
authors: Z. Yuan
image: /img/2025/0219.webp
tags: [python, google-drive, download]
description: 小檔案，大檔案，都可以下載。
---

我們寫了個 Python 程式，想要從 Google Drive 下載檔案，但有時候可以順利工作，有時候卻只拿到了一個莫名其妙的 HTML...？

肯定是程式有問題，我們得改一改。

<!-- truncate -->

## 為什麼會只下載到 HTML？

當我們向 Google Drive 發送 GET 請求，嘗試下載檔案時，如果檔案較小（一般低於 100MB），Google 可能會直接回應檔案內容，讓我們可以順利下載。

但如果檔案較大，Google 會在下載前顯示一個「病毒掃描提示頁面」，提醒使用者該檔案尚未經過完整的病毒掃描，並提供一個按鈕讓使用者自行確認下載。

如果你是操作瀏覽器，就可以手動點擊按鈕下載檔案，但如果是透過 Python 程式，除非有額外的機制來模擬點擊按鈕或解析 HTML 頁面上的下載連結，否則就只會下載到這個提示頁面本身的 HTML，而不是實際的檔案。

:::info
這個意料之外的 HTML 檔案大概內容長這樣：

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Google Drive - Virus scan warning</title>
    <meta http-equiv="content-type" content="text/html; charset=utf-8" />
   ... <!-- 其他 HTML 內容 -->
</html>
```

:::

## 解題思路：二次請求

在了解原因之後，我們可以用下列方式解決：

1. **第一次請求**：帶著檔案 ID 對 `https://docs.google.com/uc?export=download` 發送 GET。
2. **檢查回應**：如果 HTTP Header 裡面出現了 `content-disposition`，恭喜你，代表拿到的就是檔案本體，直接下載即可；若沒有，代表目前停留在病毒掃描提示頁，需要再發送一次請求。
3. **擷取驗證參數**：
   - **從 cookies 中擷取**：Google 可能在 cookie 中放置一個 `download_warning_xxxxx` 類似的 key，裡面就是 token。例如：`token = session.cookies.get('download_warning_xxxxx')`
   - **從 HTML 中擷取**：有時候 Google 不會把 token 放在 cookies，而是放在 HTML 的表單裡，例如：
     ```html
     <form
       id="download-form"
       action="https://drive.usercontent.google.com/download"
       method="get"
     >
       <input type="hidden" name="confirm" value="t" />
       ...
     </form>
     ```
     這時候可以用 [**BeautifulSoup**](https://www.crummy.com/software/BeautifulSoup/) 把所有 hidden 欄位抓下來，包含 `confirm` 或 `uuid` 等參數。
4. **組合二次請求**：拿到 `token` 後，把它帶回請求中，或直接把表單的 `action` URL 和對應的 hidden 參數全帶上，再發一次請求，就能真正開始下載。

## 程式碼實作

先安裝必要套件：

```bash
pip install requests tqdm beautifulsoup4
```

然後使用下列 Python 函式，傳入檔案 ID、要保存的檔名，以及下載後的目標資料夾，即可自動判斷是否需要二次請求並下載成功：

```python title="download_from_google.py"
import os
import re
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def download_from_google(file_id: str, file_name: str, target: str = "."):
    """
    Downloads a file from Google Drive, handling potential confirmation tokens for large files.

    Args:
        file_id (str):
            The ID of the file to download from Google Drive.
        file_name (str):
            The name to save the downloaded file as.
        target (str, optional):
            The directory to save the file in. Defaults to the current directory (".").

    Raises:
        Exception: If the download fails or the file cannot be created.

    Notes:
        This function handles both small and large files. For large files, it automatically processes
        Google's confirmation token to bypass warnings about virus scans or file size limits.

    Example:
        Download a file to the current directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt"
            )

        Download a file to a specific directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt",
                target="./downloads"
            )
    """
    # 第一次嘗試：docs.google.com/uc?export=download&id=檔案ID
    base_url = "https://docs.google.com/uc"
    session = requests.Session()
    params = {
        "export": "download",
        "id": file_id
    }
    response = session.get(base_url, params=params, stream=True)

    # 如果已經出現 Content-Disposition，代表直接拿到檔案
    if "content-disposition" not in response.headers:
        # 先嘗試從 cookies 拿 token
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        # 如果 cookies 沒有，就從 HTML 解析
        if not token:
            soup = BeautifulSoup(response.text, "html.parser")
            # 常見情況：HTML 裡面有一個 form#download-form
            download_form = soup.find("form", {"id": "download-form"})
            if download_form and download_form.get("action"):
                # 將 action 裡的網址抓出來，可能是 drive.usercontent.google.com/download
                download_url = download_form["action"]
                # 收集所有 hidden 欄位
                hidden_inputs = download_form.find_all(
                    "input", {"type": "hidden"})
                form_params = {}
                for inp in hidden_inputs:
                    if inp.get("name") and inp.get("value") is not None:
                        form_params[inp["name"]] = inp["value"]

                # 用這些參數去重新 GET
                # 注意：原本 action 可能只是相對路徑，這裡直接用完整網址
                response = session.get(
                    download_url, params=form_params, stream=True)
            else:
                # 或者有些情況是直接在 HTML 裡 search confirm=xxx
                match = re.search(r'confirm=([0-9A-Za-z-_]+)', response.text)
                if match:
                    token = match.group(1)
                    # 帶上 confirm token 再重新請求 docs.google.com
                    params["confirm"] = token
                    response = session.get(
                        base_url, params=params, stream=True)
                else:
                    raise Exception("無法在回應中找到下載連結或確認參數，下載失敗。")

        else:
            # 直接帶上 cookies 抓到的 token 再打一次
            params["confirm"] = token
            response = session.get(base_url, params=params, stream=True)

    # 確保下載目錄存在
    os.makedirs(target, exist_ok=True)
    file_path = os.path.join(target, file_name)

    # 開始把檔案 chunk 寫到本地，附帶進度條
    try:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, "wb") as f, tqdm(
            desc=file_name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"File successfully downloaded to: {file_path}")

    except Exception as e:
        raise Exception(f"File download failed: {e}")
```

## 使用方式

假設我們有一個檔案 ID：`YOUR_FILE_ID`，要下載為 `big_model.onnx` 並存在 `./models` 目錄下，只需要這樣呼叫：

```python
download_from_google(
    file_id="YOUR_FILE_ID",
    file_name="big_model.onnx",
    target="./models"
)
```

完成後，即可在 `./models` 資料夾下看到成功下載的 `big_model.onnx`，並且能在命令列上看到下載進度。

## 指令列工具

或是你希望可以直接用指令列來操作，那我們可以加一段程式來包裝一下：

```python title="download_from_google_cli.py"
from download_from_google import download_from_google
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download files from Google Drive.")
    parser.add_argument("--file-id", required=True, help="Google Drive file ID.")
    parser.add_argument("--file-name", required=True, help="Output file name.")
    parser.add_argument("--target", default=".", help="Output directory. Defaults to current folder.")
    args = parser.parse_args()

    download_from_google(file_id=args.file_id, file_name=args.file_name, target=args.target)

if __name__ == "__main__":
    main()
```

將上述程式碼儲存為 `download_from_google_cli.py`，就可以直接在命令列執行：

```bash
python download_from_google_cli.py \
 --file-id YOUR_FILE_ID \
 --file-name big_model.onnx \
 --target ./models
```

沒有意外的話，它就會自動開始下載檔案並顯示進度條。

我們測試了 70MB 和 900MB 的檔案，都能正常下載，至於 900GB 的檔案...（🤔 🤔 🤔）

沒試過，我們手邊也沒有這麼大的檔案，改天有遇到再來更新吧！
