---
sidebar_position: 3
---

# 進階安裝

## 常用參考資料

- 由 NVIDIA 建置的 PyTorch 映像的每個版本的細節，請查閱：[**PyTorch Release Notes**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

- NVIDIA runtime 前準備，請參考：[**Installation (Native GPU Support)**](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>)

- NVIDIA Toolkit 安裝方式，請參考：[**Installing the NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- ONNXRuntime 相關內容，請參考：[**ONNX Runtime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

## 環境安裝

我們的工作環境雖然不算複雜，但也是會遇到一些套件相容性的問題。

簡單來說，平常大概會分成：

- **訓練環境**：PyTorch、OpenCV、CUDA、cuDNN 要互相配合。
- **部署環境**：ONNXRuntime、OpenCV、CUDA 要互相配合。

其中，最常發生衝突的就是 PyTorch-CUDA 和 ONNXRuntime-CUDA 的版本問題。

:::tip
怎麼它們老是對不上呢？ 💢 💢 💢
:::

## 用 Docker 吧！

我們自己一律透過 docker 進行安裝，以確保環境的一致性，沒有例外。

使用 docker 可以節省大量調整環境的時間，並且可以避免許多不必要的問題。

相關環境我們在開發中也會持續測試，你只要使用以下指令：

### 安裝訓練環境

```bash
cd DocsaidKit
bash docker/build.bash
```

在「訓練環境」中，我們使用 `nvcr.io/nvidia/pytorch:24.05-py3` 作為基底映像檔。

使用者可以根據自己的需求進行更換，其中後面的編號（如：24.05）會隨時間更新。

映像檔的詳細內容請參考：[**PyTorch Release Notes**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

如果要搭配你的訓練模組，則可以在編譯完成後，再把 `docsaid_training_base_image` 作為基底映像檔，進行二次開發。

:::tip
訓練時通常不會需要用到 ONNXRuntime，就算遇到 CUDA 問題，ONNXRuntime 也能自己切換成 CPU 的模式運行。
:::

### 安裝推論環境

```bash
cd DocsaidKit
bash docker/build_infer.bash
```

在「推論環境」中，我們使用 `nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` 作為基底映像檔。

這個映像檔是專門用來部署模型的，所以不會有訓練環境的套件，你不會在裡面看到像是 PyTorch 這類的套件。

使用者可以根據自己的需求進行更換，相關版本會隨著 ONNXRuntime 的更新而有所變動。

關於用於推論系列的映像檔，可以參考：[**NVIDIA NGC**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda)

## 使用方式

一般來說，我們會把這個模組搭配像是 `DocAligner` 這類的專案進行應用。

### 日常使用

以下我們寫個範例，假設你有一個 `your_scripts.py` 的檔案，我們需要用 python 來執行這個檔案。

假設你已經完成推論環境的安裝，接著我們另外寫一個 `Dockerfile`：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM docsaid_infer_image:latest

# 設置工作目錄，使用者可以根據自己的需求進行更換
WORKDIR /code

# 舉例：安裝 DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENTRYPOINT ["python"]
```

然後建立這個映像檔：

```bash
docker build -f your_Dockerfile -t your_image_name .
```

完成後，每次使用的時候就把指令包在 docker 裡面執行：

```bash
#!/bin/bash
docker run \
    --gpus all \
    -v ${PWD}:/code \
    -it --rm your_image_name your_scripts.py
```

這樣就相當於直接調用包裝好的 python 環境，並且可以確保環境的一致性。

:::tip
如果你希望可以進到容器裡面，而不要啟動 Python，那麼可以把入口點改成 `/bin/bash`。

```Dockerfile
ENTRYPOINT ["/bin/bash"]
```

:::

### 引入 gosu 配置

如果你在執行 docker 的時候，遇到了權限問題：

- **例如：在容器中輸出檔案或影像，其權限都是 root:root，要修改和刪除都很麻煩！**

那麼我們會建議你可以考慮使用 `gosu` 這個工具。

基於 `gosu` 的使用方式，我們將原本的 Dockerfile 修改如下：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM docsaid_infer_image:latest

# 設置工作目錄，使用者可以根據自己的需求進行更換
WORKDIR /code

# 舉例：安裝 DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

# 設置入口點脚本路徑
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# 安裝 gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# 創建入口點腳本
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    fi\n\
    \n\
    # 檢查是否有參數\n\
    if [ $# -gt 0 ]; then\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# 賦予權限
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# 入口點
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

然後建立這個映像檔：

```bash
docker build -f your_Dockerfile -t your_image_name .
```

完成後，每次使用的時候就把指令包在 docker 裡面執行：

```bash
#!/bin/bash
docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    -v ${PWD}:/code
    -it --rm your_image_name your_scripts.py
```

### 安裝內部套件

如果在建置映像檔的時候，需要安裝一些內部套件，那我們需要另外帶入環境變數。

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM docsaid_infer_image:latest

# 設置工作目錄，使用者可以根據自己的需求進行更換
WORKDIR /code

# 舉例：安裝 DocAligner（假設為內部套件）

# 引入環境變數
ARG PYPI_ACCOUNT
ARG PYPI_PASSWORD

# 更改為你的內部套件源
ENV SERVER_IP=192.168.100.100:28080/simple/

# 安裝 docaligner
# 要記得更改為你的套件伺服器位址
RUN python -m pip install \
    --trusted-host 192.168.100.100 \
    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner

ENTRYPOINT ["python"]
```

然後建立這個映像檔：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_ACCOUNT=your_account \
    --build-arg PYPI_PASSWORD=your_password \
    -t your_image_name .
```

如果你的帳號密碼寫在其他地方，例如在 `pip.conf` 檔案中，也可以透過解析字串的方式來引入，例如：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \
    -t your_image_name .
```

完成後，每次使用的時候就把指令包在 docker 裡面執行，使用方式和上面一樣。

## 常見問題

### Permission denied

使用 gosu 切換使用者之後，你的權限會限縮在一定的範圍內，這時如果你需要對容器內的檔案進行讀寫，可能會遇到權限問題。

舉例來說：如果你安裝了 `DocAligner` 套件，這個套件會在啟動模型時自動下載模型檔案，並放在 python 相關的資料夾中。

在上述這個範例中，模型檔案預設存放路徑會在：

- `/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt`

這個路徑顯然已經超出了使用者的權限範圍！

所以你需要在啟動容器的時候，把這個路徑授予給使用者，請修改上面的 Dockerfile，如下：

```Dockerfile title="your_Dockerfile" {28}
# syntax=docker/dockerfile:experimental
FROM docsaid_infer_image:latest

# 設置工作目錄，使用者可以根據自己的需求進行更換
WORKDIR /code

# 舉例：安裝 DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

# 設置入口點脚本路徑
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# 安裝 gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# 創建入口點腳本
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    chmod -R 777 /usr/local/lib/python3.10/dist-packages\n\
    fi\n\
    \n\
    # 檢查是否有參數\n\
    if [ $# -gt 0 ]; then\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# 賦予權限
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# 入口點
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

如果遇到其他類似的問題，也可以透過這個方式來解決。
