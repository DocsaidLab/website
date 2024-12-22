---
sidebar_position: 3
---

# 進階安裝

## 常用參考資料

在開始進行環境建置前，有幾份官方文件十分值得參考：

- **PyTorch Release Notes**

  由 NVIDIA 提供的 [PyTorch 映像的發佈版本記錄](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)，能幫助你了解特定映像內建的 PyTorch、CUDA、cuDNN 版本，減少相依套件的衝突。

---

- **NVIDIA runtime 前置作業**：

  若想在 docker 中使用 GPU，請先參考 [Installation (Native GPU Support)](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>)，確保本機已正確安裝 NVIDIA 驅動與容器工具。

---

- **NVIDIA Container Toolkit 安裝**

  關於 [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 的官方教學，建議你詳細閱讀，這對 docker 的 GPU 加速必不可少。

---

- **ONNXRuntime Release Notes**

  在使用 ONNXRuntime 進行推論時，若需要 GPU 加速可參考官方給的 [CUDA Execution Provider 說明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)，以確保版本相容性。

## 環境安裝

大部分以深度學習為主的專案都會碰到套件相依的狀況。常見的分工方式為：

- **訓練環境**：PyTorch、OpenCV、CUDA、cuDNN 版本需相互匹配，否則常常會碰到「某個函式庫無法正確載入」的問題。
- **部署環境**：ONNXRuntime、OpenCV、CUDA 同樣需要對應合適版本，特別是 GPU 加速時，更要小心 ONNXRuntime-CUDA 對於 CUDA 版本的要求。

其中最容易踩坑的就是 **PyTorch-CUDA** 與 **ONNXRuntime-CUDA** 的版本不一致。遇到這種情形，通常建議先退回官方測試過的組合，或仔細查看它們對於 CUDA、cuDNN 的相依關係。

:::tip
怎麼它們老是對不上呢？ 💢 💢 💢
:::

## 用 Docker 吧！

為了確保一致性與可移植性，我們**強烈建議**使用 docker。若你已在本機建立環境，同樣可行，但往後在協同開發與部署階段，就會花更多時間處理無謂的衝突。

### 安裝環境

```bash
cd Capybara
bash docker/build.bash
```

建置用的 `Dockerfile` 也放在專案中，有興趣的話可以參考：[**Capybara Dockerfile**](https://github.com/DocsaidLab/Capybara/blob/main/docker/Dockerfile)

在「推論環境」中，我們使用 `nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` 作為基底映像檔。

這個映像檔是專門用來部署模型的，所以不會有訓練環境的套件，你不會在裡面看到像是 PyTorch 這類的套件。

使用者可以根據自己的需求進行更換，相關版本會隨著 ONNXRuntime 的更新而有所變動。

關於用於推論系列的映像檔，可以參考：[**NVIDIA NGC**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda)

## 使用方式

以下示範一個常見的使用場景：透過 docker 執行外部腳本，並於容器中掛載當前目錄。

### 日常使用

假設你有一個 `your_scripts.py`，想使用推論容器裡的 Python 執行，步驟如下：

1. 建立一個新的 `Dockerfile`（命名為 `your_Dockerfile`）：

   ```Dockerfile title="your_Dockerfile"
   # syntax=docker/dockerfile:experimental
   FROM capybara_infer_image:latest

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

2. 建立映像檔：

   ```bash
   docker build -f your_Dockerfile -t your_image_name .
   ```

3. 撰寫執行腳本（例如 `run_in_docker.sh`）：

   ```bash
   #!/bin/bash
   docker run \
       --gpus all \
       -v ${PWD}:/code \
       -it --rm your_image_name your_scripts.py
   ```

4. 執行 `run_in_docker.sh`，即可進行推論。

:::tip
若想進入容器並啟動 bash，而非直接跑 Python，可將 `ENTRYPOINT ["python"]` 改為 `ENTRYPOINT ["/bin/bash"]`。
:::

### 引入 gosu 配置

在實務上常會遇到「容器內輸出的檔案屬性為 root」的情況。

若多位工程師共享同一目錄，可能造成後續權限修改的麻煩。

這時，可以透過 `gosu` 解決，我們修改 Dockerfile 範例如下：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

# 舉例：安裝 DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

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

RUN chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

#### 映像建置與執行

1. 建置映像：

   ```bash
   docker build -f your_Dockerfile -t your_image_name .
   ```

2. 執行映像（以 GPU 加速為例）：

   ```bash
   #!/bin/bash
   docker run \
       -e USER_ID=$(id -u) \
       -e GROUP_ID=$(id -g) \
       --gpus all \
       -v ${PWD}:/code \
       -it --rm your_image_name your_scripts.py
   ```

這樣輸出的檔案就會自動帶有當前使用者的權限，方便後續讀寫。

### 安裝內部套件

假設需要安裝**私有套件**或**內部工具**（例如放在私有 PyPI），可在建置時帶入認證資訊：

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

ARG PYPI_ACCOUNT
ARG PYPI_PASSWORD

# 指定你的內部套件源
ENV SERVER_IP=192.168.100.100:28080/simple/

RUN python -m pip install \
    --trusted-host 192.168.100.100 \
    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner

ENTRYPOINT ["python"]
```

然後在 build 指令中，帶入帳密：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_ACCOUNT=your_account \
    --build-arg PYPI_PASSWORD=your_password \
    -t your_image_name .
```

若帳密存放於 `pip.conf`，也可以透過解析字串的方式來引入，例如：

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \
    -t your_image_name .
```

完成後，每次使用的時候就把指令包在 docker 裡面執行，使用方式和上面一樣。

## 常見問題：權限不足。

指令列顯示：Permission denied，真是人生一大憾事。

使用 gosu 切換使用者之後，你的權限會限縮在一定的範圍內，這時如果你需要對容器內的檔案進行讀寫，可能會遇到權限問題。

舉例來說：如果你安裝了 `DocAligner` 套件，這個套件會在啟動模型時自動下載模型檔案，並放在 python 相關的資料夾中。

在上述這個範例中，模型檔案預設存放路徑會在：

- `/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt`

這個路徑顯然已經超出了使用者的權限範圍！

所以你需要在啟動容器的時候，把這個路徑授予給使用者，請修改上面的 Dockerfile，如下：

```Dockerfile title="your_Dockerfile" {23}
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

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
    if [ $# -gt 0 ]; then\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

RUN chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

如果是特定目錄需授權，可只修改對應的路徑，避免過度開放權限。

## 總結

雖然使用 docker 會需要更多學習成本，但它能夠確保環境的一致性，並且在部署與協同開發時，能夠大大減少不必要的麻煩。

這個投資絕對物超所值，希望你也能享受到這個便利！
