---
slug: npm-npx-nvm-python-analogues-zh
title: Python 與 JS 的工具對應
authors: Zephyr
tags: [npm, npx, nvm, pip, pyenv]
---

你可能擅長使用 Python，你也知道 `pip`、`python -m` 之類的指令。

你可能擅長使用 JavaScript，你也知道 `npm`、`npx`、`nvm` 之類的指令。

但是換個地方就容易換個腦袋，所以我來幫你對應一下這些指令。

<!--truncate-->

## npm 與 pip

:::tip
**它們都是：套件管理器**
:::

npm（Node Package Manager）和 pip 本質上服務於相同的目的：它們分別為 Node.js 和 Python 的包管理器。包管理器對於共享、重用和管理代碼庫或模塊至關重要。

- **安裝包**：在 npm 中，你會使用 `npm install <package-name>` 命令來為你的項目添加一個庫。同樣地，pip 通過 `pip install <package-name>` 實現相同的目標。
- **版本管理**：npm 通過 `package.json` 文件追蹤包版本，確保開發團隊的每個成員使用相同版本的庫。Pip 則依賴於 `requirements.txt` 或較新工具如 pipenv 和 poetry 來實現類似的功能。
- **發布包**：npm 使開發者能夠將他們的包發布到 npm 註冊處，使其可供全球 Node.js 社區使用。Pip 透過 PyPI（Python Package Index）提供此能力，允許分享 Python 包。

## npx 與 -m 標誌

:::tip
**它們都是：直接執行命令的工具**
:::

npx（npm package runner）和 Python 的 `-m` 標誌解決了直接在終端執行包命令的需求，無需進行全局安裝。

- **直接執行**：npx 允許你直接從命令行執行項目本地 `node_modules` 文件夾中安裝的任何包（或者如果沒有安裝，則從 npm 註冊處獲取），Python 透過 `-m` 標誌實現類似結果，允許直接執行模塊，例如使用 `python -m http.server` 命令啟動一個簡單的 HTTP 服務器。

:::note
**npm run 與 npx run**

- npm run：在 JavaScript 項目中，npm run 用於執行 package.json 文件中定義的腳本。這是一種執行項目特定任務（如測試、構建或部署）的常用方法。
- npx run：雖然 npx 通常用於執行單個命令或包，但它主要用於直接執行未全局安裝的包。npx run 不是一個標準命令，npx 的常見用法不包括 run 關鍵字，而是直接跟隨包名或命令。
:::

## nvm、pyenv 與 conda

:::tip
**它們都是：版本管理工具**
:::

沒有合適的工具，切換不同版本的 Node.js 或 Python 可能會非常麻煩。nvm（Node Version Manager）、pyenv 和 conda 為此問題提供了解決方案，允許開發者在同一台機器上安裝並切換 Node.js 或 Python 的多個版本。

- **版本切換**：nvm 使用諸如 `nvm use <version>` 的命令來切換 Node.js 版本。Pyenv 和 conda 為 Python 提供了類似的功能，pyenv 通過 `pyenv global <version>` 或 `pyenv local <version>` 來進行版本切換，而 conda 則使用 `conda activate <environment-name>` 切換到不同的環境，每個環境可以擁有不同的 Python 版本和包。
- **多版本管理**：這些工具促進了在同一台機器上管理多個版本，解決了因版本差異可能導致的衝突問題。

## 結論

當我在學習新的語言或框架時，我總是喜歡將其與我已經熟悉的工具進行對應。這樣做有助於我更快地理解新工具的功能和用法，並且幫助我更快地上手。

希望這篇文章能幫助你更好地理解 npm、npx、nvm、pip 和 pyenv 之間的對應關係，並且幫助你更好地使用這些工具。