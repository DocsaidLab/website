---
sidebar_position: 2
---

# 基本安裝

在開始安裝 Capybara 之前，請確保你的系統符合以下要求：

## 依賴套件

請依照作業系統，安裝下列必要的系統套件：

- **Ubuntu**

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  ```bash
  brew install jpeg-turbo exiftool ffmpeg
  ```

  - **特別注意**：經過測試，在 macOS 上使用 libheif 時，存在一些已知問題，主要包括：

    1. **生成的 HEIC 檔案無法打開**：在 macOS 上，libheif 生成的 HEIC 檔案可能無法被某些程式打開。這可能與圖像尺寸有關，特別是當圖像的寬度或高度為奇數時，可能會導致相容性問題。

    2. **編譯錯誤**：在 macOS 上編譯 libheif 時，可能會遇到與 ffmpeg 解碼器相關的未定義符號錯誤。這可能是由於編譯選項或相依性設定不正確所致。

    3. **範例程式無法執行**：在 macOS Sonoma 上，libheif 的範例程式可能無法正常運行，出現動態鏈接錯誤，提示找不到 `libheif.1.dylib`，這可能與動態庫的路徑設定有關。

    由於問題不少，因此我們目前只在 Ubuntu 才會運行 libheif，至於 macOS 的部分則留給未來的版本。

### pdf2image

pdf2image 是用於將 PDF 文件轉換成影像的 Python 模組，請確保系統已安裝下列工具：

- MacOS：需要安裝 poppler

  ```bash
  brew install poppler
  ```

- Linux：大多數發行版已內建 `pdftoppm` 與 `pdftocairo`。如未安裝，請執行：

  ```bash
  sudo apt install poppler-utils
  ```

### ONNXRuntime

若需使用 ONNXRuntime 進行 GPU 加速推理，請確保已安裝相容版本的 CUDA，如下示範：

```bash
sudo apt install cuda-12-4
# 假設要加入至 .bashrc
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```

## 透過 PyPI 安裝

1. 透過 PyPI 安裝套件：

   ```bash
   pip install capybara-docsaid
   ```

2. 驗證安裝：

   ```bash
   python -c "import capybara; print(capybara.__version__)"
   ```

3. 若顯示版本號，則安裝成功。

## 透過 git clone 安裝

1. 下載本專案：

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

4. 安裝建置完成的 wheel 檔：

   ```bash
   pip install dist/capybara_docsaid-*-py3-none-any.whl
   ```

## 常見問題

1. **為什麼沒有支援 Windows 的安裝？**

   珍愛生命，遠離 Windows。

---

2. **我就想用 Windows，我勸你別多管閒事！**

   好吧，我們建議你安裝 Docker，然後使用上述的方法，透過 Docker 來執行你的程式。

   請參考下一篇：[**進階安裝**](./advance.md)。

---

3. **Docker 怎麼裝？**

   不難，但步驟有點多。

   請參考 [**Docker 官方文件**](https://docs.docker.com/get-docker/) 進行安裝。
