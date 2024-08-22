---
slug: convert-pdf-to-images
title: 使用 Python 把 PDF 轉圖片
authors: Zephyr
tags: [Python, pdf2image]
image: /img/2024/0214.webp
description: 使用開源套件 pdf2image 來解決問題。
---

<figure>
![title](/img/2024/0214.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

在日常辦公或學習中，無論是為了更便捷地分享資訊，還是將文件內容整合進演示文稿中，我們經常需要將 PDF 檔案轉換成圖片格式。

這裡我們推薦一個好用的 Python 模組： [pdf2image](https://github.com/Belval/pdf2image/tree/master) ，它能夠將 PDF 檔案轉換成 PIL 圖片。

這篇教學將引導你如何安裝並使用這個套件。

<!-- truncate -->

## 安裝依賴

`pdf2image` 依賴於 `pdftoppm` 和 `pdftocairo`，不同作業系統的安裝方式略有不同：

- **Mac**：通過 Homebrew 安裝 Poppler：`brew install poppler`。
- **Linux**：大多數 Linux 發行版已預裝 `pdftoppm` 和 `pdftocairo`。若未安裝，請透過包管理器安裝 `poppler-utils`。
- **使用 `conda`**：無論哪個平台，都可以使用 `conda` 安裝 Poppler：`conda install -c conda-forge poppler`，然後再安裝 `pdf2image`。

## 安裝 `pdf2image`

首先，你需要安裝 `pdf2image`，在終端機中輸入以下指令即可安裝：

```shell
pip install pdf2image
```

## 使用 `pdf2image` 轉換 PDF

轉換 PDF 至圖片的基本用法非常簡單：

```python
from pdf2image import convert_from_path

images = convert_from_path('/path/to/your/pdf/file.pdf')
```

這將把 PDF 的每一頁轉換成一個 PIL 圖片對象，並儲存在 `images` 列表中。

你也可以從二進制數據轉換 PDF：

```python
images = convert_from_bytes(open('/path/to/your/pdf/file.pdf', 'rb').read())
```

## 可選參數

`pdf2image` 提供了豐富的可選參數，允許你自定義 DPI、輸出格式、頁面範圍等。例如：使用 `dpi=300` 提高輸出圖片的清晰度，或者使用 `first_page` 和 `last_page` 指定轉換範圍。

你可以參考 `pdf2image` 的[官方文件](https://github.com/Belval/pdf2image/tree/master)；或是參考我們自己改寫的 [pdf2imgs](https://github.com/DocsaidLab/DocsaidKit/blob/eb8ac0a56779a75dcc951c683001e6129052cc5a/docsaidkit/vision/improc.py#L275) 函數來了解更多用法。

## 結語

`pdf2image` 是一個功能強大且易於使用的工具，能夠滿足你將 PDF 轉換為圖片的需求。無論是用於文檔處理、資料整理，還是內容展示，它都能提供高效的解決方案。

希望這篇教學能夠幫助你輕鬆掌握 `pdf2image` 的使用，提高你的工作與學習效率。
