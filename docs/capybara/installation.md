---
sidebar_position: 2
---

# 基本安裝

在開始安裝 Capybara 之前，請確保你的系統符合以下要求：

## 前置條件

### Python 版本

- 確保系統已安裝 Python 3.10 或以上版本。

### 依賴套件

根據你的作業系統，安裝所需的依賴套件。

- **Ubuntu**

  開啟終端，執行以下命令安裝依賴：

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  使用 brew 安裝相依性：

  ```bash
  brew install jpeg-turbo exiftool ffmpeg libheif
  ```

### pdf2image 依賴套件

pdf2image 是一個 Python 模組，用於將 PDF 文件轉換為圖片。

根據你的作業系統，請遵循以下指示進行安裝：

- 或參考開源專案 [**pdf2image**](https://github.com/Belval/pdf2image) 相關頁面以取得安裝指南。

- MacOS：Mac 使用者需要安裝 poppler。透過 Brew 進行安裝：

  ```bash
  brew install poppler
  ```

- Linux：大多數 Linux 發行版已預裝 `pdftoppm` 和 `pdftocairo`。

  如果未安裝，請透過你的套件管理器安裝 poppler-utils。

  ```bash
  sudo apt install poppler-utils
  ```

## 安裝套件

滿足前提條件後，你可以透過 git clone 安裝：

1. 下載本套件：

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. 安裝 wheel 套件：

   ```bash
   pip install wheel
   ```

3. 建構 wheel 檔案：

   ```bash
   cd Capybara
   python setup.py bdist_wheel
   ```

4. 安裝建置的 wheel 套件：

   ```bash
   pip install dist/capybara-*-py3-none-any.whl
   ```

## 常見問題

1. **為什麼沒有 Windows？**

   珍愛生命，遠離 Windows。

2. **我就想用 Windows，我勸你別多管閒事！**

   好吧，我們建議你安裝 Docker，然後使用上述的方法，透過 Docker 來執行你的程式。

   請參考下一篇：[**進階安裝**](./advance.md)。

3. **Docker 怎麼裝？**

   不難，但步驟有點多。

   請參考 [**Docker 官方文件**](https://docs.docker.com/get-docker/) 進行安裝。
