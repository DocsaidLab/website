---
slug: error-record
title: 日常錯誤排除紀錄
authors: Z. Yuan
tags: [error, record]
image: /img/2024/0204.webp
description: 紀錄一些簡單問題和解法。
---

寫程式總是會遇到一堆問題。

我們在這裡紀錄一些瑣碎的問題和解決方法。

:::tip
本文章會持續更新。
:::

<!-- truncate -->

## 1. 執行 `npx docusaurus start` 時出現以下錯誤

- **錯誤訊息：**

  ```bash
  file:///home/user/workspace/blog/node_modules/@docusaurus/core/bin/docusaurus.mjs:30
  process.env.BABEL_ENV ??= 'development';
                      ^^^

  SyntaxError: Unexpected token '??='
  ```

- **解決方法：**

  `??=` 操作符需要 Node.js 15.0.0 或更高版本才能支持。

  ```bash
  nvm install node
  nvm use node
  ```

## 2. choco 命令無法辨識

- **錯誤訊息：**

  ```shell
  PS C:\Windows\System32> choco install git -y
  >>
  choco : 無法辨識 'choco' 詞彙是否為 Cmdlet、函數、指令檔或可執行程式的名稱。請檢查名稱拼字是否正確，如果包含路徑的話，請確認路徑是否正確，然後再試一次。
  位於 線路:1 字元:1
  + choco install git -y
  + ~~~~~
      + CategoryInfo          : ObjectNotFound: (choco:String) [], CommandNotFoundException
      + FullyQualifiedErrorId : CommandNotFoundException
  ```

- **解決方法：**

  這表示沒有成功安裝 Chocolatey，失敗的原因通常是沒有以「系統管理員」身份執行 PowerShell。

  請以「系統管理員」身份執行 PowerShell，然後再次執行 Chocolatey 安裝指令。

  ```shell
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```

## 3. Chocolatey 安裝失敗

- **錯誤訊息：**

  ```shell
  警告: An existing Chocolatey installation was detected. Installation will not continue. This script will not overwrite existing installations.
  If there is no Chocolatey installation at 'C:\ProgramData\chocolatey', delete the folder and attempt the installation again.

  Please use choco upgrade chocolatey to handle upgrades of Chocolatey itself.
  If the existing installation is not functional or a prior installation did not complete, follow these steps:
  - Backup the files at the path listed above so you can restore your previous installation if needed.
  - Remove the existing installation manually.
  - Rerun this installation script.
  - Reinstall any packages previously installed, if needed (refer to the lib folder in the backup).

  Once installation is completed, the backup folder is no longer needed and can be deleted.
  ```

- **解決方法：**

  這表示 Chocolatey 已經安裝過了，請先刪除舊的安裝，然後再重新安裝。

  ```shell
  Remove-Item "C:\ProgramData\chocolatey" -Recurse -Force
  ```

## 4. 遠端機器埠口轉發

- **描述：**

  在遠端機器上啟動了一個服務，例如 TensorBoard，但是無法直接訪問，因此必須通過本地機器進行轉發。

- **解決方法：**

  假設服務運行在遠端機器的 6006 端口，本地機器想要訪問的端口也是 6006。

  我們在使用 SSH 登入時，可以通過 `-L` 參數 進行端口轉發：

  ```bash
  ssh -L 6006:localhost:6006 user@remote_ip_address
  ```

  這樣本地機器就可以通過 `http://localhost:6006` 訪問遠端機器的 TensorBoard 服務了。

## 5. 開發和部署環境網頁渲染行為不一致

- **描述：**

  在 `custom.css` 中設置了部落格的版面樣式：

  ```css
  .container {
    max-width: 90%;
    padding: 0 15px;
    margin: 0 auto;
  }
  ```

  在部署階段，這個樣式似乎被其他更高優先級的樣式覆蓋了，但是在開發階段，這個樣式是正常的。

- **解決方法：**

  更具體地選擇目標：

  ```css
  body .container {
    max-width: 90%;
    padding: 0 15px;
    margin: 0 auto;
  }
  ```

## 6. Turbojpg 讀取影像出現警告

- **描述**

  讀取影像時出現以下警告訊息：

  ```shell
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: 18 extraneous bytes before marker 0xc4
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: bad Huffman code
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: premature end of data segment
  ```

- **解決方法**

  看了很煩，應該要過濾剔除這些資料：

  ```python
  import cv2
  import warnings

  data = ['test1.jpg', 'test2.jpg', 'test3.jpg']

  for d in data:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always", UserWarning)

      # 讀取影像，看看是否有警告
      cv2.imread(d)

      # 有的話就刪除
      if w:
        data.remove(d)

  # 最後可以透過 json 或其他方式紀錄清洗後的資料
  ```

## 7. `Docusaurus` 部署後 `showLastUpdateTime: true` 無效

- **描述**

  在 `docusaurus.config.js` 中設置了 `showLastUpdateTime: true` 和 `showLastUpdateAuthor: true,`，但是部署後發現沒有效果，渲染頁面中全部都是一樣的時間和作者？

- **解決方法**

  因為在部署時，checkout 分支的時候設定錯誤，導致 `git` 無法正確獲取最後更新時間和作者。

  這樣改：

  ```yaml
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
  ```

  只要設定 `fetch-depth: 0`，問題就解決了。
