---
slug: python-js-basic-command-equivalents
title: Python 與 JS 的基本指令對應
authors: Zephyr
image: /img/2024/0407.webp
tags: [npm, pip]
description: 把 Python 與 JS 的基本指令進行類似的對應。
---

<figure>
![title](/img/2024/0407.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

我比較常使用的語言是 Python，其中常見的指令像是 `pip` 和 `python -m`，但最近開始學習 JavaScript 時，發現了類似功能的指令，像是 `npm`、`npx` 和 `nvm` 等。

我想試著對應這些指令，或許能更容易地轉換新技能。

<!-- truncate -->

## npm 與 pip

:::tip
**它們都是：套件管理器**
:::

npm（Node Package Manager）和 pip 本質上服務於相同的目的：它們分別為 Node.js 和 Python 的包管理器。

- **安裝包**：在 npm 中，我們使用 `npm install <package-name>` 命令來為項目添加一個庫。同樣地，pip 通過 `pip install <package-name>` 實現相同的目標。
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

切換不同版本的 Node.js 或 Python 可能會非常麻煩。

nvm（Node Version Manager）、pyenv 和 conda 為此問題提供了解決方案，允許開發者在同一台機器上安裝並切換 Node.js 或 Python 的多個版本。

- **版本切換**：nvm 使用諸如 `nvm use <version>` 的命令來切換 Node.js 版本。Pyenv 和 conda 為 Python 提供了類似的功能，pyenv 通過 `pyenv global <version>` 或 `pyenv local <version>` 來進行版本切換，而 conda 則使用 `conda activate <environment-name>` 切換到不同的環境，每個環境可以擁有不同的 Python 版本和包。
- **多版本管理**：這些工具促進了在同一台機器上管理多個版本，解決了因版本差異可能導致的衝突問題。

## 題外話：還有 yarn 呢？

npm 於 2009 年誕生，是 Node.js 生態系統中首個主要的包管理器。npm 的出現解決了 JavaScript 開發者管理依賴包的需求，迅速成為行業標準。然而，隨著使用者數量的增長，npm 暴露出了一些性能問題，如安裝速度慢和依賴版本不穩定。

Yarn 於 2016 年由 Facebook 開發，目的是在解決 npm 的性能瓶頸。Yarn 引入了並行下載、離線快取等技術，大幅提升了依賴安裝速度和穩定性。此外，Yarn 還增加了 yarn.lock 文件，確保依賴包版本的一致性。這些改進使得 Yarn 迅速獲得了開發者的青睞。

隨著 Yarn 的成功，npm 也加快了自身的改進步伐。從 npm v5 開始，npm 引入了 package-lock.json，提升了安裝速度並增強了依賴管理的穩定性。至今，npm 和 Yarn 已經成為 JavaScript 開發者的兩大主要選擇，並且都在不斷進步，互相借鑒對方的優點。

## 結論

當我在學習新的語言或框架時，我總是喜歡將其與我已經熟悉的工具進行對應。這樣做有助於我更快地理解新工具的功能和用法，並且幫助我更快地上手。

這些指令可能不能百分之百地對應，但它們確實有一些相似之處，加減能幫上一點忙吧。
