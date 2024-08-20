---
slug: pyenv-installation
title: 使用 pyenv 管理 Python 版本
authors: Zephyr
tags: [python, pyenv, virtualenv, usage]
image: /img/2023/1010.webp
description: 記錄安裝 pyenv 的安裝與使用方式。
---

<figure>
![title](/img/2023/1010.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

早些年使用 Python 的時候大多使用 Conda 來管理。現在則是常用 pyenv 。

我們就在這篇文章中來記錄安裝 pyenv 的安裝與使用方式。

<!-- truncate -->

## 前置條件

在安裝 `pyenv` 之前，你需要有 `Git` 安裝在你的系統上。

:::info
在 pyenv 套件中有提供 [**安裝問題指南**](https://github.com/pyenv/pyenv/wiki/Common-build-problems) 。

如果你在安裝過程中遇到問題，可以參考這個頁面。
:::

## 安裝 `pyenv`

1. **執行安裝指令**：

   你可以通過以下指令快速安裝 `pyenv`：

   ```bash
   curl https://pyenv.run | bash
   ```

   這條指令實際上是從 GitHub 上的 `pyenv-installer` 存儲庫獲取安裝腳本並執行。

2. **設定你的 Shell 環境**：

   安裝完畢後，根據 [**設定指南**](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv) 配置你的 shell 環境，以確保 `pyenv` 能夠正確工作。

   如果你是使用 `bash`，則需要將以下代碼添加到你的 `.bashrc` 文件中：

   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

   如果你使用的是 `zsh`，則需要將上述代碼添加到你的 `.zshrc` 文件中。

3. **重啟你的 Shell**：

   當你完成上述步驟後，請重新載入新的配置。

   ```bash
   exec $SHELL
   ```

## 使用 `pyenv`

安裝和設定完成後，你可以開始使用 `pyenv` 管理多個 Python 版本：

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

在 Python 開發中，虛擬環境是一個非常重要的概念。

它可以幫助我們在不同的專案中使用不同的 Python 版本和依賴庫。

至少當你不小心弄壞了 Python 環境時，你可以直接刪除虛擬環境，重新來過。

:::tip
我們一律建議在開發 Python 專案時使用虛擬環境。
:::

### 安裝

`pyenv` 還提供了一個 `pyenv-virtualenv` 插件，可以讓你更方便地管理 Python 虛擬環境。

在早期這個功能需要單獨安裝，現在已經整合到 `pyenv` 中，我們可以直接用：

```bash
pyenv virtualenv 3.10.14 your-env-name
```

其中，`3.10.14` 是你想要使用的 Python 版本，你在上一個步驟已經完成安裝，`your-env-name` 是虛擬環境的名稱。

### 使用

激活虛擬環境，請運行：

```bash
pyenv activate your-env-name
```

### 移除

最後，當你不需要虛擬環境，可以運行以下命令刪除：

```bash
pyenv virtualenv-delete your-env-name
```

## 更新 `pyenv`

如果需要更新 `pyenv` 到最新版本，只需運行：

```bash
pyenv update
```

## 卸載 `pyenv`

如果你決定不再使用 `pyenv`，可以按照以下步驟卸載：

1. **移除 `pyenv` 安裝目錄**：

   ```bash
   rm -fr ~/.pyenv
   ```

2. **清理你的 `.bashrc`**：
   移除或註釋掉相關的 `pyenv` 配置行，然後重啟你的 shell：
   ```bash
   exec $SHELL
   ```
