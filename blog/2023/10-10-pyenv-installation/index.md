---
slug: pyenv-installation
title: 使用 pyenv 管理 Python 版本
authors: Z. Yuan
tags: [pyenv, virtualenv]
image: /img/2023/1010.webp
description: 記錄安裝 pyenv 的安裝與使用方式。
---

早些年使用 Python 時大多依賴 Conda 進行管理，如今則以 pyenv 成為常用工具。

在本篇文章中，我簡單記錄如何安裝與使用 pyenv，並針對不同作業系統提供必要的補充說明。

<!-- truncate -->

## 前置條件

在安裝 `pyenv` 之前，你需要有 `Git` 安裝在你的系統上。

:::info
在 pyenv 套件中有提供 [**安裝問題指南**](https://github.com/pyenv/pyenv/wiki/Common-build-problems) 。

如果你在安裝過程中遇到問題，可以參考這個頁面。
:::

## 常見問題與解決方案

以下整理幾個重要案例與解決方案：

- **依賴套件不足**
  請先依照 [**pyenv 官方建議的依賴環境**](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) 安裝所有必要的套件與 build 工具。

- **zlib 擴展編譯失敗**

  錯誤訊息常見為：

  - `ERROR: The Python zlib extension was not compiled. Missing the zlib?`

  解決方法：
  - 在 Ubuntu/Debian 系統上，安裝 `zlib1g` 與 `zlib1g-dev` 等相關套件：
    ```bash
    sudo apt install zlib1g zlib1g-dev
    ```
  - 在 macOS 上，若以 Homebrew 安裝 zlib，可設定環境變數：
    ```bash
    CPPFLAGS="-I$(brew --prefix zlib)/include" pyenv install -v <python版本>
    ```

- **OpenSSL 擴展編譯失敗**

  若出現

  - `ERROR: The Python ssl extension was not compiled. Missing the OpenSSL lib?`

  解決方法：

  - 確認已安裝 OpenSSL 開發套件（例如 Ubuntu 使用 `sudo apt install libssl-dev`，Fedora 使用 `sudo dnf install openssl-devel`）。
  - 如 OpenSSL 安裝在非標準路徑，則設定：
    ```bash
    CPPFLAGS="-I<openssl安裝路徑>/include" \
    LDFLAGS="-L<openssl安裝路徑>/lib" \
    pyenv install -v <python版本>
    ```

- **系統資源不足**

  出現「resource temporarily unavailable」錯誤時，可嘗試降低 make 的並行數量：

  ```bash
  MAKE_OPTS='-j 1' pyenv install <python版本>
  ```

- **python-build 定義未找到**

  當遇到 `python-build: definition not found` 錯誤，請更新 python-build 定義：

  ```bash
  cd ~/.pyenv/plugins/python-build && git pull
  ```

- **macOS 架構相關錯誤**

  若遇到類似 `ld: symbol(s) not found for architecture x86_64` 或 `ld: symbol(s) not found for architecture arm64` 的錯誤，請確認 Homebrew 套件是否對應正確架構，並檢查是否需設定額外的環境變數（如 CPPFLAGS、LDFLAGS 及 CONFIGURE_OPTS）。

更多詳細資訊請參考 [**Common build problems**](https://github.com/pyenv/pyenv/wiki/Common-build-problems) 。

## 跨平台注意事項

- **Linux/macOS：**
  - 安裝方式基本相同，可直接使用後續指令。
  - 請依作業系統安裝必要的編譯相依庫（例如 Ubuntu 可能需要安裝 `build-essential`、`libssl-dev`、`zlib1g-dev` 等）。

- **Windows 使用者：**
  - pyenv 原生設計為 Unix-like 環境，建議使用 [**pyenv-win**](https://github.com/pyenv-win/pyenv-win) 版本。
  - 或者，可在 Windows 下使用 WSL、Git Bash 等工具以獲得類 Unix 的操作環境。

- **其他 Shell 使用者：**
  - 若你使用非 bash 或 zsh 的 shell（如 fish），請參考該 shell 的設定文件進行相應調整。

## 安裝 `pyenv`

1. **執行安裝指令**：

   你可以透過以下指令快速安裝 `pyenv`：

   ```bash
   curl https://pyenv.run | bash
   ```

   該指令會從 GitHub 上的 `pyenv-installer` 存儲庫取得安裝腳本並執行。

2. **設定你的 Shell 環境**：

   安裝完成後，根據 [**設定指南**](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv) 配置你的 shell 環境，以確保 `pyenv` 正常運作。

   若使用 `bash`，請將以下代碼添加至你的 `.bashrc` 文件中：

   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

   若你使用 `zsh`，請將相同代碼添加至 `.zshrc` 文件；其他 shell 請參考相應設定文件。

3. **重啟你的 Shell**：

   完成上述步驟後，重新載入配置：

   ```bash
   exec $SHELL
   ```

## 使用 `pyenv`

安裝及設定完成後，你即可使用 `pyenv` 管理多個 Python 版本：

- **安裝新的 Python 版本**：

  ```bash
  pyenv install 3.10.14
  ```

- **切換全局 Python 版本**：

  ```bash
  pyenv global 3.10.14
  ```

- **在特定目錄使用特定版本**：

  ```bash
  pyenv local 3.8.5
  ```

## 虛擬環境

在 Python 開發中，虛擬環境非常重要，可協助你在不同專案中使用獨立的 Python 版本及依賴庫，避免環境衝突。

:::tip
我個人會建議在每個 Python 專案中都使用虛擬環境，即使在不慎損壞環境時，也可輕鬆刪除並重建。
:::

### 安裝

`pyenv` 提供了 `pyenv-virtualenv` 插件，讓虛擬環境管理更為方便。

該功能現已整合至 `pyenv` 中，可直接使用：

```bash
pyenv virtualenv 3.10.14 your-env-name
```

其中，`3.10.14` 為欲使用的 Python 版本（請先確認已安裝），`your-env-name` 為虛擬環境名稱。

### 使用

啟動虛擬環境：

```bash
pyenv activate your-env-name
```

### 移除

當不再需要虛擬環境時，可透過下列指令移除：

```bash
pyenv virtualenv-delete your-env-name
```

## 更新 `pyenv`

若需更新 `pyenv` 至最新版本，請參考以下方式：

- **使用更新插件：** 若已安裝 [**pyenv-update**](https://github.com/pyenv/pyenv-update) 插件，可直接執行：

  ```bash
  pyenv update
  ```

- **手動更新：**
  進入 `~/.pyenv` 目錄後，使用 Git 指令更新：

  ```bash
  cd ~/.pyenv
  git pull
  ```

## 移除 `pyenv`

若決定不再使用 `pyenv`，請依下列步驟移除：

1. **移除 `pyenv` 安裝目錄**：

   ```bash
   rm -fr ~/.pyenv
   ```

2. **清理 Shell 配置**：

   移除或註解掉 `.bashrc`、`.zshrc`（或其他 shell 配置文件）中與 `pyenv` 相關的配置行，然後重新啟動 shell：

   ```bash
   exec $SHELL
   ```

## 小結

常用的指令大概就是這樣，祝你有個美好的 Python 環境。