---
sidebar_position: 2
---

# Installation

在開始安裝 DocsaidKit 之前，請確保您的系統符合以下要求：

## 前置條件

### Python 版本

- 確保系統已安裝 Python 3.8 或以上版本。

### 依賴套件

根據您的作業系統，安裝所需的依賴套件。

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

### pdf2image 安裝

- 請參考開源專案 [**pdf2image**](https://github.com/Belval/pdf2image) 相關頁面以取得安裝指南。

## 安裝套件

滿足前提條件後，您可以選擇以下方法之一進行安裝：

### 一般方法

本包僅透過我們的內部 Pypi 服務提供。

外部使用者應透過 git 複製並使用 setup.py 安裝。

### 透過 git clone 安裝

1. 安裝 wheel 套件：

    ```bash
    pip install wheel
    ```

2. 建構 wheel 檔案：

    ```bash
    python setup.py bdist_wheel
    ```

3. 安裝建置的 wheel 套件：

    ```bash
    pip install dist/docsaidKit-${version}-none-any.whl
    ```

    如果需要安裝支援 PyTorch 的版本：

    ```bash
    pip install "dist/docsaidKit-${version}-none-any.whl[torch]"
    ```

### 透過 docker 安裝（建議）

我們一律透過 docker 進行安裝，以確保環境的一致性，沒有例外。

所以我們也同樣建議您使用 docker 進行安裝。

```bash
bash docker/build.bash
```

## 常見問題

1. **依賴安裝**

    特定的依賴套件如 `libturbojpeg`, `exiftool`, `ffmpeg`, `libheif-dev` 在不同作業系統上的安裝方法可能會有所不同。 上述指南提供了在 Ubuntu 和 MacOS 上安裝這些依賴的命令。 這些依賴對於確保 DocsaidKit 中的特定功能正常運作是必要的。
