---
sidebar_position: 2
---

# 安裝

在開始安裝 DocsaidKit 之前，請確保你的系統符合以下要求：

## 前置條件

### Python 版本

- 確保系統已安裝 Python 3.8 或以上版本。

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

滿足前提條件後，你可以選擇以下方法之一進行安裝：

### 透過 git clone 安裝

1. 下載本套件：

    ```bash
    git clone https://github.com/DocsaidLab/DocsaidKit.git
    ```

2. 安裝 wheel 套件：

    ```bash
    pip install wheel
    ```

3. 建構 wheel 檔案：

    ```bash
    cd DocsaidKit
    python setup.py bdist_wheel
    ```

4. 安裝建置的 wheel 套件：

    ```bash
    pip install dist/docsaidkit-*-py3-none-any.whl
    ```

    如果需要安裝支援 PyTorch 的版本：

    ```bash
    pip install "dist/docsaidKit-${version}-none-any.whl[torch]"
    ```

### 透過 docker 安裝（建議）

我自己是一律透過 docker 進行安裝，以確保環境的一致性，沒有例外。

所以我也同樣建議你使用 docker 進行安裝，相關環境我都測試好了，你只要使用以下指令：

```bash
cd DocsaidKit
bash docker/build.bash
```

完成後，每次使用的時候就把指令包在 docker 裡面執行：

```bash
docker run -v ${PWD}:/code -it docsaid_training_base_image your_scripts.py
```

以下是 Build Script 具體內容，每個版本都會調整部分細節：

以下內容是基於 DocsaidKit V0.18.0。

```dockerfile title="DocsaidKit/docker/Dockerfile"
# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/pytorch:24.02-py3

# 這確保在運行 apt 命令時不會有任何用戶交互。
# 這是為了確保 Docker 映像建構過程自動進行，無需人工介入。
ENV DEBIAN_FRONTEND=noninteractive

# 防止 Python 創建 .pyc 字節碼文件。
ENV PYTHONDONTWRITEBYTECODE=1

# 為Matplotlib和Transformers建立配置和緩存目錄
RUN mkdir -p /app/matplotlib_config /app/transformers_cache

# 設置環境變量
ENV MPLCONFIGDIR /app/matplotlib_config
ENV TRANSFORMERS_CACHE /app/transformers_cache

# 確保目錄具有正確的權限
RUN chmod -R 777 /app/matplotlib_config /app/transformers_cache

# 安裝 tzdata 套件並設定時區為 Asia/Taipei
RUN apt update -y && apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
ENV TZ=Asia/Taipei

# 安裝基本套件
# libturbojpeg -> 用於 JPEG 圖像的庫
# exiftool -> 用於讀取圖像 EXIF 資訊
# ffmpeg -> 處理音頻和視頻的工具
# poppler-utils -> 用於 PDF 轉換成影像
# libpng-dev -> 用於處理 PNG 圖像的庫
# libtiff5-dev -> 用於處理 TIFF 圖像的庫
# libjpeg8-dev -> 用於處理 JPEG 圖像的庫
# libopenjp2-7-dev -> 用於處理 JPEG 2000 圖像的庫
# zlib1g-dev -> 用於壓縮和解壓縮的庫
# libfreetype6-dev -> 用於處理 TrueType 和 OpenType 字體的庫
# liblcms2-dev -> 用於處理色彩管理系統的庫
# libwebp-dev -> 用於處理 WebP 圖像的庫
# tcl8.6-dev -> GUI 工具包
# tk8.6-dev -> GUI 工具包
# python3-tk -> GUI 工具包
# libharfbuzz-dev -> 用於處理 Unicode 文本的庫
# libfribidi-dev -> 用於處理 Unicode 文本的庫
# libxcb1-dev -> X 協議 C-language Binding庫
# libfftw3-dev -> 用於處理快速傅立葉轉換的庫
RUN apt update -y && apt upgrade -y && apt install -y git \
    libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U setuptools pip wheel

# For training
RUN python -m pip install -U tqdm colored ipython tabulate \
    tensorboard scikit-learn fire albumentations Pillow>=10.0.0 fitsne

# 安裝 docsaidkit
COPY . /usr/local/DocsaidKit
RUN cd /usr/local/DocsaidKit && python setup.py bdist_wheel && \
    python -m pip install $(ls dist/*.whl | sed 's/$/[torch]/') && \
    rm -rf /usr/local/DocsaidKit

# Fixed 4.8.0.76 import error
RUN python -m pip install opencv-python==4.8.0.74

# Preload data
RUN python -c "import docsaidkit"

WORKDIR /code

CMD ["bash"]
```


## 常見問題

1. **為什麼沒有 Windows？**

    抱歉，我對 Windows 環境建置有 PTSD（創傷後壓力症候群），所以沒有支援。

    珍愛生命，遠離 Windows。

2. **我就想用 Windows，我勸你別多管閒事！**

    好吧，我建議你安裝 Docker，然後使用 Docker Image 來執行你的程式。

2. **Docker 怎麼裝？**

    不難，但步驟有點多。

    請參考 [**Docker 官方文件**](https://docs.docker.com/get-docker/) 進行安裝。
