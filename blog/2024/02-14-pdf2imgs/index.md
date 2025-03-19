---
slug: convert-pdf-to-images
title: 使用 Python 把 PDF 轉圖片
authors: Z. Yuan
tags: [Python, pdf2image]
image: /img/2024/0214.webp
description: 使用開源套件 pdf2image 來解決問題。
---

開發過程中，你可能經常需要將 PDF 檔案轉換成圖片格式，無論是用於文件展示、資料處理或內容分享。

本篇文章介紹一個好用的 Python 模組： [**pdf2image**](https://github.com/Belval/pdf2image/tree/master) ，它能夠將 PDF 檔案轉換成 PIL 圖片。

<!-- truncate -->

## 安裝依賴

`pdf2image` 依賴於 `pdftoppm` 和 `pdftocairo` 兩個工具，不同作業系統的安裝方式略有不同：

- **Mac**：透過 Homebrew 安裝 Poppler，請在終端機中執行：

  ```shell
  brew install poppler
  ```
- **Linux**：大多數 Linux 發行版已預裝 `pdftoppm` 和 `pdftocairo`。如果沒有，可以用以下指令：

  ```shell
  sudo apt-get install poppler-utils   # Ubuntu/Debian 系統
  ```
- **使用 `conda`**：無論哪個平台，都可以透過 `conda` 安裝 Poppler：

  ```shell
  conda install -c conda-forge poppler
  ```

  安裝完成後，再安裝 `pdf2image` 即可。

## 安裝 `pdf2image`

在終端機中執行以下指令即可完成安裝：

```shell
pip install pdf2image
```

## 使用方式

轉換 PDF 至圖片的基本用法相當簡單。

以下範例示範如何將 PDF 的每一頁轉換成 PIL 圖片對象，並儲存成檔案：

```python
from pdf2image import convert_from_path

# 將 PDF 檔案轉換為圖片列表
images = convert_from_path('/path/to/your/pdf/file.pdf')

# 逐頁儲存為 PNG 格式的圖片
for i, image in enumerate(images):
    image.save(f'output_page_{i+1}.png', 'PNG')
```

若你希望從二進制數據進行轉換，可以參考以下做法：
```python
with open('/path/to/your/pdf/file.pdf', 'rb') as f:
    pdf_data = f.read()

images = convert_from_bytes(pdf_data)
```

## 可選參數與進階設定

`pdf2image` 提供了豐富的可選參數，讓你能根據需求自定義輸出圖片的品質與範圍：

- **DPI 設定**：調整 `dpi` 參數可以提升圖片解析度，適用於需要高品質圖片的場合：

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', dpi=300)
  ```

- **指定頁面範圍**： 使用 `first_page` 與 `last_page` 參數，可選擇僅轉換特定頁面：

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', first_page=2, last_page=5)
  ```

- **輸出圖片格式**： 可透過 `fmt` 參數指定輸出圖片的格式，如 JPEG 或 PNG：

  ```python
  images = convert_from_path('/path/to/your/pdf/file.pdf', fmt='jpeg')
  ```

- **錯誤處理**：在轉換過程中，可能會遇到格式錯誤或檔案損毀的情況，建議搭配 try/except 捕捉異常：

  ```python
  try:
      images = convert_from_path('/path/to/your/pdf/file.pdf')
  except Exception as e:
      print("轉換失敗：", e)
  ```

## 結語

`pdf2image` 是個好用的工具，更多參數與詳細用法，請參考 [**pdf2image 官方文件**](https://github.com/Belval/pdf2image/tree/master)。
