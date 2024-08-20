---
slug: error-record
title: 日常錯誤排除紀錄
authors: Zephyr
tags: [error, record]
image: /img/2024/0204.webp
description: 紀錄一些簡單問題和解法。
---

<figure>
![title](/img/2024/0204.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

我們總是會遇到一堆問題。有些問題是我們自己造成的，有些問題是別人造成的，有些問題是我們無法控制的。這裡紀錄一些簡單問題和解決方法。

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
